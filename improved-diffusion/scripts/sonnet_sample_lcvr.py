"""Sonnet unconditional sampling + LCVR decoding pipeline.

流程：
  1. 读 sonnet diffusion ckpt
  2. unconditional p_sample 生成 (B, T, C) latent
  3. 对每个 target_son10.json 中的 sonnet：
       构造 LCVR position mask
         - lt_positions    -> {< (id)}
         - eos_positions   -> {eos (id)}
         - gt_positions    -> {> (id)}
         - tail (i > last_gt) -> {_PAD}
         - 其他 word slot -> vocab \ {_PAD,_GO,_EOS,_UNK,<,eos,>}
       对每个 latent 应用 mask + argmax
  4. 同一批 latent 的 baseline 解码（无 mask argmax）也保留作为对照
  5. 输出 json：{baseline:[...], lcvr:[...]} for 10 targets × 50 cands

用法：
  python -u scripts/sonnet_sample_lcvr.py \
    --ckpt diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355/ema_0.9999_025000.pt \
    --target_json control_gen/target_son10.json \
    --num_samples 50 --batch_size 25 \
    --out out_gen/sonnet_subset10.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch as th

# noinspection PyUnresolvedReferences
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from improved_diffusion import dist_util, logger
from improved_diffusion.script_util import (
    create_model_and_diffusion,
    args_to_dict,
    model_and_diffusion_defaults,
)
from improved_diffusion.rounding import load_models
from improved_diffusion.test_util import get_weights, denoised_fn_round


SPECIAL_BASE = {"START", "END", "UNK", "PAD"}
LINE_TRIPLET = {"<", "eos", ">"}


def build_vocab(model_dir: Path):
    vpath = model_dir / "vocab.json"
    if not vpath.exists():
        raise FileNotFoundError(f"no vocab at {vpath}")
    with open(vpath, "r", encoding="utf-8") as f:
        v = json.load(f)
    inv = {i: w for w, i in v.items()}
    return v, inv


def build_position_masks(record, T: int, vocab: dict, V_logits: int):
    """对单条 sonnet target 构造 (T, V_logits) bool mask。

    vocab 实际词数 |vocab| 通常 ≤ V_logits（embedding padding）。
    超出 |vocab| 的 logit 索引全部 forbidden。
    """
    V_real = len(vocab)
    mask = th.zeros(T, V_logits, dtype=th.bool)
    pad_id = vocab["PAD"]
    lt_id = vocab.get("<")
    eos_word_id = vocab.get("eos")
    gt_id = vocab.get(">")
    forbidden_in_word = {
        vocab[t] for t in (SPECIAL_BASE | LINE_TRIPLET) if t in vocab
    }
    # 只允许 [0, V_real) 之间的 word token
    word_mask_row = th.zeros(V_logits, dtype=th.bool)
    word_mask_row[:V_real] = True
    for tid in forbidden_in_word:
        word_mask_row[tid] = False

    last_triplet_end = max(record["gt_positions"])

    fixed = {}
    for p in record["lt_positions"]:
        fixed[p] = lt_id
    for p in record["eos_positions"]:
        fixed[p] = eos_word_id
    for p in record["gt_positions"]:
        fixed[p] = gt_id

    for i in range(T):
        if i in fixed:
            tid = fixed[i]
            if tid is not None:
                mask[i, tid] = True
        elif i > last_triplet_end:
            mask[i, pad_id] = True
        else:
            mask[i] = word_mask_row
    return mask


def decode_with_lcvr(model, latents, position_mask, device):
    """latents: (B, T, C) torch tensor on device.
    position_mask: (T, V) bool tensor on device.
    return: ids (B, T) long on cpu.
    """
    with th.no_grad():
        logits = model.get_logits(latents)  # (B, T, V)
    pmask = position_mask.unsqueeze(0)  # (1, T, V)
    masked = logits.masked_fill(~pmask, float("-inf"))
    ids = masked.argmax(dim=-1)
    return ids.cpu()


def decode_baseline(model, latents):
    """Argmax without LCVR."""
    with th.no_grad():
        logits = model.get_logits(latents)
    ids = logits.argmax(dim=-1)
    return ids.cpu()


def ids_to_tokens(ids: th.Tensor, inv_vocab: dict):
    out = []
    for row in ids.tolist():
        out.append([inv_vocab.get(i, f"<{i}>") for i in row])
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target_json", required=True)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=25)
    p.add_argument("--use_ddim", action="store_true")
    p.add_argument("--diff_respacing", default=None,
                   help="override timestep_respacing (e.g. 'ddim25')")
    p.add_argument("--clamp", default="clamp")
    p.add_argument("--top_p", type=float, default=-1.0)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    th.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.ckpt)
    model_dir = ckpt_path.parent
    config_path = model_dir / "training_args.json"
    with open(config_path, "r") as f:
        training_args = json.load(f)

    # patch args
    # 先用 model_and_diffusion_defaults() 给一组完整默认，再覆盖 training_args
    defaults = model_and_diffusion_defaults()
    merged = dict(defaults)
    merged.update(training_args)
    cls = argparse.Namespace(**merged)
    cls.batch_size = args.batch_size
    cls.sigma_small = True
    cls.clamp = args.clamp
    cls.top_p = args.top_p
    if args.diff_respacing is not None:
        cls.timestep_respacing = args.diff_respacing
    if not hasattr(cls, "clip_denoised"):
        cls.clip_denoised = False
    if not hasattr(cls, "model_name_or_path"):
        cls.model_name_or_path = ""
    if not hasattr(cls, "experiment"):
        cls.experiment = "random"
    if cls.experiment == "random1":
        cls.experiment = "random"

    dist_util.setup_dist()
    logger.configure()

    print(f"[info] loading model: {ckpt_path}", flush=True)
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(cls, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(dist_util.load_state_dict(str(ckpt_path), map_location="cpu"))
    device = dist_util.dev()
    model.to(device)
    model.eval()

    model2, tokenizer = load_models(
        cls.modality, cls.experiment, cls.model_name_or_path, cls.in_channel, str(model_dir)
    )
    if cls.training_mode.startswith("e2e"):
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    vocab, inv_vocab = build_vocab(model_dir)
    print(f"[info] vocab size = {len(vocab)}", flush=True)

    # 1. unconditional sampling
    T = cls.image_size ** 2
    C = cls.in_channel
    print(f"[info] sampling B={args.num_samples} T={T} C={C}", flush=True)
    sample_fn = diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop

    model3 = get_weights(model2, cls)
    all_latents = []
    while sum(x.shape[0] for x in all_latents) < args.num_samples:
        bs = min(args.batch_size, args.num_samples - sum(x.shape[0] for x in all_latents))
        sample = sample_fn(
            model,
            (bs, T, C),
            clip_denoised=cls.clip_denoised,
            denoised_fn=partial(denoised_fn_round, cls, model3.to(device)) if cls.clamp == "clamp" else None,
            model_kwargs={},
            top_p=cls.top_p,
        )
        all_latents.append(sample)
        print(f"  got {sum(x.shape[0] for x in all_latents)}/{args.num_samples}", flush=True)

    latents = th.cat(all_latents, dim=0)[: args.num_samples]  # (N, T, C)
    print(f"[info] total latents: {latents.shape}", flush=True)

    # 2. load targets
    targets = []
    with open(args.target_json, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            targets.append(json.loads(line))
    print(f"[info] {len(targets)} targets", flush=True)

    # 3. baseline argmax (target-independent)
    baseline_ids = decode_baseline(model, latents)
    baseline_tokens = ids_to_tokens(baseline_ids, inv_vocab)

    # 4. LCVR per target
    # 推断 logits 维度
    with th.no_grad():
        sample_logits = model.get_logits(latents[:1])
    V_logits = sample_logits.shape[-1]
    print(f"[info] V_logits (model) = {V_logits}, |vocab| = {len(vocab)}", flush=True)

    out = {"baseline": baseline_tokens, "lcvr": {}}
    for rec in targets:
        idx = rec["subset_idx"]
        pmask = build_position_masks(rec, T, vocab, V_logits).to(device)
        lcvr_ids = decode_with_lcvr(model, latents, pmask, device)
        out["lcvr"][str(idx)] = {
            "src_idx": rec["src_idx"],
            "n_tokens": rec["n_tokens"],
            "tokens": ids_to_tokens(lcvr_ids, inv_vocab),
        }
        print(f"  [{idx}] LCVR done; first cand head: {out['lcvr'][str(idx)]['tokens'][0][:8]}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
