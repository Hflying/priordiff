"""E2E NLG: unconditional sampling + LCVR decoding.

E2E 的 LCVR 模板比 ci/sonnet 更简单（form-independent baseline）：
  - 词位 (i < L)         : V_i = vocab \ {START, END, UNK, PAD}
  - 边界 (i == L)        : V_i = {END}（若 vocab 有 END）
  - 尾部 (i > L)         : V_i = {PAD}

L 来自 target 真值长度，演示 LCVR 即使没有显式 punctuation 模板也能保证
结构（无 special token 漏出、END 在边界上、tail 全 PAD）。

输出 json：{baseline:[...], lcvr: {idx: {tokens:[...], n_tokens:int, src_idx:int}}}

用法：
  python -u scripts/e2e_sample_lcvr.py \
    --ckpt diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_e2e/ema_0.9999_025000.pt \
    --target_jsonl control_gen/target_e2e30.jsonl \
    --num_samples 50 --batch_size 50 \
    --out out_gen/e2e_e2e30_25000.json
"""

from __future__ import annotations
import argparse
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch as th

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


def build_vocab(model_dir: Path):
    vpath = model_dir / "vocab.json"
    if not vpath.exists():
        raise FileNotFoundError(f"no vocab at {vpath}")
    with open(vpath, "r", encoding="utf-8") as f:
        v = json.load(f)
    inv = {i: w for w, i in v.items()}
    return v, inv


def build_position_masks(L: int, T: int, vocab: dict, V_logits: int):
    """L: target true length; T: total positions."""
    V_real = len(vocab)
    mask = th.zeros(T, V_logits, dtype=th.bool)
    pad_id = vocab["PAD"]
    end_id = vocab.get("END")
    forbidden = {vocab[t] for t in SPECIAL_BASE if t in vocab}

    word_row = th.zeros(V_logits, dtype=th.bool)
    word_row[:V_real] = True
    for tid in forbidden:
        word_row[tid] = False

    for i in range(T):
        if i < L:
            mask[i] = word_row
        elif i == L and end_id is not None:
            mask[i, end_id] = True
        else:
            mask[i, pad_id] = True
    return mask


def decode_with_lcvr(model, latents, position_mask):
    with th.no_grad():
        logits = model.get_logits(latents)
    pmask = position_mask.unsqueeze(0)
    masked = logits.masked_fill(~pmask, float("-inf"))
    return masked.argmax(dim=-1).cpu()


def decode_baseline(model, latents):
    with th.no_grad():
        logits = model.get_logits(latents)
    return logits.argmax(dim=-1).cpu()


def ids_to_tokens(ids: th.Tensor, inv_vocab: dict):
    return [[inv_vocab.get(int(i), f"<{int(i)}>") for i in row] for row in ids.tolist()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target_jsonl", required=True)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--use_ddim", action="store_true")
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

    defaults = model_and_diffusion_defaults()
    merged = dict(defaults)
    merged.update(training_args)
    cls = argparse.Namespace(**merged)
    cls.batch_size = args.batch_size
    cls.sigma_small = True
    cls.clamp = args.clamp
    cls.top_p = args.top_p
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
    model.to(device).eval()

    model2, _ = load_models(
        cls.modality, cls.experiment, cls.model_name_or_path, cls.in_channel, str(model_dir)
    )
    if cls.training_mode.startswith("e2e"):
        model2.weight = th.nn.Parameter(model.word_embedding.weight.clone().cpu())

    vocab, inv_vocab = build_vocab(model_dir)
    print(f"[info] vocab size = {len(vocab)}", flush=True)

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

    latents = th.cat(all_latents, dim=0)[: args.num_samples]
    print(f"[info] total latents: {tuple(latents.shape)}", flush=True)

    targets = []
    with open(args.target_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    print(f"[info] {len(targets)} targets", flush=True)

    with th.no_grad():
        sample_logits = model.get_logits(latents[:1])
    V_logits = sample_logits.shape[-1]
    print(f"[info] V_logits = {V_logits}, |vocab| = {len(vocab)}", flush=True)

    baseline_ids = decode_baseline(model, latents)
    baseline_tokens = ids_to_tokens(baseline_ids, inv_vocab)

    out = {"baseline": baseline_tokens, "lcvr": {}}
    for rec in targets:
        idx = str(rec["subset_idx"])
        L = rec["n_tokens"]
        pmask = build_position_masks(L, T, vocab, V_logits).to(device)
        lcvr_ids = decode_with_lcvr(model, latents, pmask)
        out["lcvr"][idx] = {
            "src_idx": rec["src_idx"],
            "n_tokens": L,
            "slots": rec["slots"],
            "value_tokens": rec["value_tokens"],
            "tokens": ids_to_tokens(lcvr_ids, inv_vocab),
        }
        print(f"  [{idx}] L={L} LCVR done; cand0 head: {out['lcvr'][idx]['tokens'][0][:8]}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
