"""M3-v0: LCVR + Repetition Penalty (training-free cascaded refinement).

LCVR 已经做到的：100% 结构正确，但 head 字位会反复选「野马」「归」这种锚字。

M3-v0 思路：在 LCVR 的 argmax 上加一层 **token-level repetition penalty** —
对当前候选序列已经出现 ≥ k 次的 token 给一个 logit 惩罚，让第二/三选挤上来。

形式化：对位置 i，
  argmax_i = argmax_{w ∈ V_i} ( logit[i, w] - α * count_so_far(w) )

其中 count_so_far 是 (a) 该候选其它位置已选 token 的全局计数 (sequence-level)，
  (b) 可选 sliding-window 局部计数。

实现：iterative — 一次性算所有位置的 argmax 不会引入 dependency，
所以我们做 K 轮 Gibbs 风格 update：
  1. 初始化 = LCVR argmax
  2. 每轮按位置遍历，用 logit - α * current_count 重新选
  3. 直到收敛或达到 K

用法：
  cd improved-diffusion
  python scripts/decode_pkl_m3_v0.py \
      --model_path diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/ema_0.9999_400000.pt \
      --pkl out_gen/control_tone_vowel_length/checkpoint_tone_vowel_length_sz12_subset10_step400k_bfreeze.pkl \
      --target_json control_gen/target_ci_subset10.json \
      --corpus ../datasets/ci8w/ci_train.txt \
      --restrict_freq drop_low_zero \
      --rep_alpha 4.0 \
      --rep_iters 5 \
      --out out_gen/control_tone_vowel_length/infill_m3v0_400k_bfreeze.json
"""

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch as th

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from improved_diffusion.rounding import load_tokenizer

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
from decode_pkl_lcvr import (  # noqa: E402
    SPECIAL_TOKENS, PUNCT_TOKENS,
    build_model, load_freq, build_position_masks,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--pkl", required=True)
    p.add_argument("--target_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus", default=None)
    p.add_argument("--restrict_freq", default="none",
                   choices=["none", "drop_zero", "drop_low_zero", "high_only"])
    p.add_argument("--rep_alpha", type=float, default=4.0,
                   help="repetition penalty 强度（每出现 1 次扣 alpha logit）")
    p.add_argument("--rep_iters", type=int, default=5,
                   help="Gibbs 迭代步数")
    p.add_argument("--rep_threshold", type=int, default=1,
                   help="只对 count ≥ threshold 才施加惩罚")
    p.add_argument("--rep_window", type=int, default=0,
                   help="若 > 0，只统计前 window 个位置内的字（局部惩罚）；0 = 全 head 区")
    return p.parse_args()


def lcvr_argmax_with_penalty(logits, position_mask, target_words, vocab,
                              alpha, threshold, window, n_iters):
    """logits: (B, T, V_full) on cuda
    position_mask: (T, V_full) bool on cuda
    target_words: list[str], len L
    返回 ids: (B, T) np.int64

    迭代：每轮 i=0..L-1 重选 token，统计 prev 已选 token 数。
    """
    B, T, V_full = logits.shape
    device = logits.device
    L = len(target_words)
    is_punct_target = th.tensor(
        [w in PUNCT_TOKENS for w in target_words] + [False] * (T - L),
        dtype=th.bool, device=device,
    )

    # 初始化：纯 LCVR argmax
    masked = logits.masked_fill(~position_mask.unsqueeze(0), float("-inf"))
    ids = masked.argmax(dim=-1)  # (B, T)

    # 在 head 区做 Gibbs（仅字位；标点位 / boundary / tail 由 LCVR 锁死）
    char_positions = [i for i in range(L) if not is_punct_target[i].item()]

    for it in range(n_iters):
        changed = 0
        for i in char_positions:
            # 计算"其余字位"在每个 batch 的字 count
            if window > 0:
                lo = max(0, i - window)
                hi = min(L, i + window + 1)
                other_idx = [j for j in range(lo, hi) if j != i and not is_punct_target[j]]
            else:
                other_idx = [j for j in char_positions if j != i]
            if not other_idx:
                continue
            other_ids = ids[:, other_idx]  # (B, n_other)

            # 构造 (B, V_full) 的 count tensor
            counts = th.zeros((B, V_full), dtype=th.float32, device=device)
            counts.scatter_add_(
                1, other_ids,
                th.ones_like(other_ids, dtype=th.float32),
            )

            # apply threshold: count > threshold 的部分计算 penalty
            penalty = th.clamp(counts - threshold, min=0.0) * alpha  # (B, V_full)

            scores = logits[:, i, :] - penalty  # (B, V_full)
            # 应用 LCVR 的位置 mask
            scores = scores.masked_fill(~position_mask[i].unsqueeze(0), float("-inf"))
            new_id = scores.argmax(dim=-1)
            changed += (new_id != ids[:, i]).sum().item()
            ids[:, i] = new_id

        if changed == 0:
            break

    return ids


def decode(args, model, tokenizer, vocab, freq):
    device = args.device if th.cuda.is_available() else "cpu"
    targets = []
    with open(args.target_json) as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))

    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    sample_dict = payload["sample_dict"]

    n = min(len(targets), len(sample_dict))
    sample_keys = list(sample_dict.keys())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoded_pairs = []
    for i in range(n):
        target = targets[i]
        latents = sample_dict[sample_keys[i]]
        T = latents.shape[1]
        pmask = build_position_masks(target["words_"], T, vocab, freq, args.restrict_freq).to(device)

        # 1. logits
        x_t = th.tensor(latents, dtype=th.float32, device=device)
        with th.no_grad():
            logits = model.get_logits(x_t)  # (B, T, V_full)
        B, _, V_full = logits.shape
        if pmask.size(1) < V_full:
            pad_mask = th.zeros((T, V_full - pmask.size(1)), dtype=th.bool, device=device)
            pmask_use = th.cat([pmask, pad_mask], dim=1)
        elif pmask.size(1) > V_full:
            pmask_use = pmask[:, :V_full]
        else:
            pmask_use = pmask

        # 2. M3-v0: penalty argmax
        ids = lcvr_argmax_with_penalty(
            logits, pmask_use, target["words_"], vocab,
            alpha=args.rep_alpha, threshold=args.rep_threshold,
            window=args.rep_window, n_iters=args.rep_iters,
        )

        words = []
        for seq in ids.cpu().numpy():
            toks = [tokenizer[int(t)] for t in seq]
            words.append(" ".join(toks))
        decoded_pairs.append((sample_keys[i], words))
        L = len(target["words_"])
        head_chars = [tokenizer[int(t)] for t in ids[0, :L].cpu().numpy() if tokenizer[int(t)] not in PUNCT_TOKENS]
        uniq_ratio = len(set(head_chars)) / max(len(head_chars), 1)
        print(f"  [{i+1}/{n}] L={L}  uniq={uniq_ratio:.2f}  cand0_head={words[0].split(' ')[:6]}")

    with open(out_path, "w", encoding="utf-8") as f:
        for k, v in decoded_pairs:
            print({k: v}, file=f)
    print(f"[done] wrote {len(decoded_pairs)} entries → {out_path}")


def main():
    args = parse_args()
    device = args.device if th.cuda.is_available() else "cpu"
    print(f"[info] device={device}")
    print(f"[info] rep_alpha={args.rep_alpha}, rep_iters={args.rep_iters}, "
          f"rep_threshold={args.rep_threshold}, rep_window={args.rep_window}")

    model, train_args = build_model(args.model_path, device)
    print(f"[info] image_size={train_args['image_size']}")

    vocab = json.load(open(os.path.join(os.path.dirname(args.model_path), "vocab.json")))
    tokenizer = load_tokenizer("e2e-tgt", "random", os.path.dirname(args.model_path))

    freq = None
    if args.corpus and args.restrict_freq != "none":
        print(f"[info] loading freq from {args.corpus}")
        freq = load_freq(args.corpus, vocab)

    decode(args, model, tokenizer, vocab, freq)


if __name__ == "__main__":
    main()
