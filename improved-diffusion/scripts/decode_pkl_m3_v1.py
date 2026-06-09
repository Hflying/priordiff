"""M3-v1: LCVR 序列 + 训练好的 BERT MLM 做重复字精炼。

流程：
  1) 从 latent 用 LCVR 解码（与 decode_pkl_lcvr 相同），或直接从 --lcvr_json 读入；
  2) 对每条候选，迭代：统计 head 字位 token 频次，对出现 ≥ --repeat_threshold 的字位
     置 [MASK]，一次前向 MLM，在 LCVR 位置 mask 内 argmax 更新这些位；
  3) 标点 / boundary / tail 位不改动。

依赖：
  - diffusion_models/m3v1_mlm_ci8w/mlm_final.pt（或 mlm_step_*.pt）
  - 同目录 config.json + m3v1_meta.json（由 train_m3v1_mlm.py 写出）

用法：
  cd improved-diffusion
  TRANSFORMERS_OFFLINE=1 python scripts/decode_pkl_m3_v1.py \\
      --diffusion_ckpt diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/ema_0.9999_400000.pt \\
      --mlm_ckpt diffusion_models/m3v1_mlm_ci8w/mlm_final.pt \\
      --pkl out_gen/.../checkpoint_...pkl \\
      --target_json control_gen/target_ci_subset10.json \\
      --corpus ../datasets/ci8w/ci_train.txt --restrict_freq drop_low_zero \\
      --out out_gen/.../infill_m3v1.json
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch as th

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "transformers" / "src"
if TRANSFORMERS_SRC.is_dir():
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import BertConfig, BertForMaskedLM  # noqa: E402

from improved_diffusion.rounding import load_tokenizer  # noqa: E402

from decode_pkl_lcvr import (  # noqa: E402
    build_model,
    build_position_masks,
    load_freq,
    decode_with_lcvr,
    PUNCT_TOKENS,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diffusion_ckpt", required=True)
    p.add_argument("--mlm_ckpt", required=True)
    p.add_argument("--pkl", default="")
    p.add_argument("--target_json", required=True)
    p.add_argument("--lcvr_json", default="",
                   help="若提供则跳过扩散解码，直接读 LCVR 行")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus", default=None)
    p.add_argument("--restrict_freq", default="drop_low_zero",
                   choices=["none", "drop_zero", "drop_low_zero", "high_only"])
    p.add_argument("--refine_iters", type=int, default=8)
    p.add_argument("--repeat_threshold", type=int, default=3)
    return p.parse_args()


def load_mlm(ckpt_path: str, device: th.device):
    ckpt = th.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta") or {}
    out_dir = Path(ckpt_path).parent
    if (out_dir / "config.json").exists():
        config = BertConfig.from_pretrained(str(out_dir))
    else:
        raise FileNotFoundError(f"expect config.json beside {ckpt_path}")
    model = BertForMaskedLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    mask_id = int(meta.get("mask_token_id", config.vocab_size - 1))
    pad_id = int(meta.get("pad_token_id", 3))
    return model, mask_id, pad_id


def text_to_ids(text: str, vocab: dict, unk_id: int, T: int, pad_id: int):
    toks = text.strip().split()
    ids = []
    for t in toks[:T]:
        ids.append(int(vocab.get(t, unk_id)))
    while len(ids) < T:
        ids.append(pad_id)
    return th.tensor(ids, dtype=th.long).unsqueeze(0)


def refine_batch(
    ids: th.Tensor,
    words_,
    pmask: th.Tensor,
    mlm: BertForMaskedLM,
    mask_id: int,
    pad_id: int,
    repeat_threshold: int,
    max_iters: int,
    punct_set,
    device: th.device,
):
    """ids: (B, T) on device. 同一 target 的多条候选批量 MLM 精炼。"""
    B, T = ids.shape
    L = len(words_)
    ids = ids.clone()
    char_positions = [i for i in range(L) if words_[i] not in punct_set]
    Vlog = mlm.config.vocab_size

    for _ in range(max_iters):
        to_fix = th.zeros(B, T, dtype=th.bool, device=device)
        for b in range(B):
            cnt = Counter(int(ids[b, i].item()) for i in char_positions)
            for i in char_positions:
                tid = int(ids[b, i].item())
                if cnt[tid] >= repeat_threshold:
                    to_fix[b, i] = True
        if not to_fix.any():
            break
        inp = ids.clone()
        inp[to_fix] = mask_id
        attn = (inp != pad_id).long()
        with th.no_grad():
            logits = mlm(input_ids=inp, attention_mask=attn).logits
        for b in range(B):
            for i in char_positions:
                if not to_fix[b, i]:
                    continue
                row = logits[b, i].clone()
                pm = pmask[i].to(device)
                if pm.size(0) < Vlog:
                    pm = th.cat(
                        [pm, th.zeros(Vlog - pm.size(0), dtype=th.bool, device=device)],
                        dim=0,
                    )
                elif pm.size(0) > Vlog:
                    pm = pm[:Vlog]
                row = row.masked_fill(~pm, float("-inf"))
                ids[b, i] = int(row.argmax().item())
    return ids


def main():
    args = parse_args()
    device = args.device if th.cuda.is_available() else "cpu"
    device = th.device(device)

    diff_dir = os.path.dirname(args.diffusion_ckpt)
    vocab = json.load(open(os.path.join(diff_dir, "vocab.json")))
    unk_id = int(vocab["UNK"])
    tokenizer = load_tokenizer("e2e-tgt", "random", diff_dir)

    print(f"[info] loading MLM from {args.mlm_ckpt}", flush=True)
    mlm, mask_id, pad_id = load_mlm(args.mlm_ckpt, device)
    print(f"[info] MLM mask_id={mask_id} pad_id={pad_id}", flush=True)

    punct_set = PUNCT_TOKENS

    targets = []
    with open(args.target_json, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))

    freq = None
    if args.corpus and args.restrict_freq != "none":
        freq = load_freq(args.corpus, vocab)

    lcvr_lines = None
    if args.lcvr_json:
        print(f"[info] loading LCVR from {args.lcvr_json}")
        import ast
        lcvr_lines = []
        with open(args.lcvr_json, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lcvr_lines.append(ast.literal_eval(line))

    if lcvr_lines is None:
        print(f"[info] loading diffusion + pkl {args.pkl}")
        diff_model, train_args = build_model(args.diffusion_ckpt, str(device))
        with open(args.pkl, "rb") as f:
            sample_dict = pickle.load(f)["sample_dict"]
        sample_keys = list(sample_dict.keys())
    else:
        diff_model = None
        sample_dict = None
        sample_keys = [None] * len(lcvr_lines)

    n = min(len(targets), len(sample_keys))
    out_pairs = []

    for i in range(n):
        target = targets[i]
        words_ = target["words_"]
        L = len(words_)

        if lcvr_lines is not None:
            d = lcvr_lines[i]
            key = next(iter(d.keys()))
            cands = d[key]
            T = len(cands[0].split())
            pmask = build_position_masks(words_, T, vocab, freq, args.restrict_freq)
        else:
            key = sample_keys[i]
            latents = sample_dict[key]
            T = latents.shape[1]
            pmask = build_position_masks(words_, T, vocab, freq, args.restrict_freq)
            cands = decode_with_lcvr(diff_model, latents, pmask, tokenizer, str(device))

        refined = []
        rows = [text_to_ids(text, vocab, unk_id, T, pad_id) for text in cands]
        ids_batch = th.cat(rows, dim=0).to(device)
        ids2 = refine_batch(
            ids_batch, words_, pmask, mlm, mask_id, pad_id,
            args.repeat_threshold, args.refine_iters, punct_set, device,
        )
        for b in range(ids2.size(0)):
            toks = [tokenizer[int(t)] for t in ids2[b].cpu().tolist()]
            refined.append(" ".join(toks))
        out_pairs.append((key, refined))
        print(f"  [{i+1}/{n}] refined cand0_head={refined[0].split()[:8]}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in out_pairs:
            print({k: v}, file=f)
    print(f"[done] wrote {len(out_pairs)} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
