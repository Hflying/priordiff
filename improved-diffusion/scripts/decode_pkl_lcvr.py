"""LCVR (Length-Conditional Voronoi Rounding) 独立解码脚本.

跟 decode_pkl.py 的差别：每个位置 i 用 target words_[i] 模板限定 vocab subset，
然后 argmax 只在该 subset 上做。这样可以彻底消除：
  1. 字位置吐 START/END/PAD/UNK（特殊 token 在 head 区漏出）
  2. 字位置吐错位标点（句长被切碎）
  3. tail 区漂浮的重复占位字

LCVR 规则：
  - 位置 i 在 target.words_ 范围内（i < L）：
      * 真值此位是标点 → 投到 {，。？！、；：「」（）}
      * 真值此位是字   → 投到「字 vocab」= 全 vocab - {标点} - {特殊}
  - 位置 i = L（border）：
      * 投到 {END}
  - 位置 i ∈ (L, T)：
      * 投到 {END, PAD}
注意：「字 vocab」中保留「频次档」选项，可只允许 high-freq 字以观察更激进的修复。

用法：
  cd improved-diffusion
  python scripts/decode_pkl_lcvr.py \
      --model_path diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/ema_0.9999_400000.pt \
      --pkl out_gen/control_tone_vowel_length/checkpoint_tone_vowel_length_sz12_subset10_step400k_bfreeze.pkl \
      --target_json control_gen/target_ci_subset10.json \
      --out out_gen/control_tone_vowel_length/infill_lcvr_400k_bfreeze.json
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


SPECIAL_TOKENS = {"START", "END", "PAD", "UNK", "STOP"}
PUNCT_TOKENS = {"，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--pkl", required=True)
    p.add_argument("--target_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--corpus", default=None,
                   help="可选：训练语料路径，用于计算字频，启用 --restrict_freq")
    p.add_argument("--restrict_freq", default="none",
                   choices=["none", "drop_zero", "drop_low_zero", "high_only"],
                   help="more restrictive vocab subset for char positions")
    p.add_argument("--mask_mode", default="lcvr_full",
                   choices=["lcvr_full", "no_pos", "lax_vocab",
                            "unconstrained"],
                   help="ablation — controls which constraints are in V_i."
                        " lcvr_full: full per-position template mask (default LCVR)."
                        " no_pos: same vocab subset for all positions"
                        " (specials+puncts dropped, no position awareness)."
                        " lax_vocab: drop specials only, puncts allowed"
                        " anywhere, still use position-aware template."
                        " unconstrained: re-round with no mask (sanity).")
    return p.parse_args()


def build_model(model_path, device):
    cfg_path = os.path.join(os.path.dirname(model_path), "training_args.json")
    with open(cfg_path) as f:
        train_args = json.load(f)
    defaults = model_and_diffusion_defaults()
    init_kwargs = {k: train_args.get(k, v) for k, v in defaults.items()}
    model, _ = create_model_and_diffusion(**init_kwargs)
    state = th.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_args


def load_freq(corpus_path, vocab):
    """返回 freq_arr (V_emb,) — 对训练语料按字符统计"""
    cnt = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 1)
            text = parts[1] if len(parts) >= 2 else parts[0]
            for ch in text:
                cnt[ch] += 1
    V_emb = max(vocab.values()) + 2  # 留 buffer 给 emb 比 vocab 多 1 行的情况
    freq = np.zeros(V_emb, dtype=np.int64)
    for tok, fid in vocab.items():
        freq[fid] = cnt.get(tok, 0)
    return freq


def build_position_masks(target_words, T, vocab, freq=None,
                          restrict_freq="none", mask_mode="lcvr_full"):
    """对 1 条 target 构造长度 T 的 vocab mask 矩阵。

    返回：(T, V) bool tensor on cpu, True = 允许投影

    mask_mode controls which ablation of V_i is in effect:
      - lcvr_full: the full per-position template mask (paper default)
      - no_pos   : same vocab subset for all positions (specials+puncts
                   dropped), no position-awareness at all
      - lax_vocab: drop specials only (puncts allowed anywhere), still
                   use position-aware template
      - unconstrained: all-ones mask (degrades to baseline argmax, for
                   sanity-checking)
    """
    L = len(target_words)
    V = max(vocab.values()) + 1
    all_ids = np.arange(V)
    special_ids = np.array([vocab[t] for t in SPECIAL_TOKENS if t in vocab],
                           dtype=np.int64)
    punct_ids = np.array([vocab[t] for t in PUNCT_TOKENS if t in vocab],
                         dtype=np.int64)

    char_mask = np.ones(V, dtype=bool)
    char_mask[special_ids] = False
    char_mask[punct_ids] = False

    if restrict_freq != "none" and freq is not None:
        if restrict_freq == "drop_zero":
            char_mask[: V] &= freq[:V] > 0
        elif restrict_freq == "drop_low_zero":
            pos_mask = freq[:V] > 0
            f_low = np.percentile(freq[:V][pos_mask], 20) if pos_mask.sum() else 0
            char_mask[:V] &= (freq[:V] > f_low)
        elif restrict_freq == "high_only":
            pos_mask = freq[:V] > 0
            f_high = np.percentile(freq[:V][pos_mask], 80) if pos_mask.sum() else 0
            char_mask[:V] &= (freq[:V] >= f_high)

    end_id = vocab.get("END", 1)
    pad_id = vocab.get("PAD", 3)

    if mask_mode == "unconstrained":
        return th.ones((T, V), dtype=th.bool)

    if mask_mode == "no_pos":
        row = np.ones(V, dtype=bool)
        row[special_ids] = False
        row[punct_ids] = False
        if restrict_freq != "none" and freq is not None:
            row &= char_mask
        masks = np.tile(row, (T, 1))
        return th.tensor(masks, dtype=th.bool)

    masks = np.zeros((T, V), dtype=bool)
    for i in range(T):
        if i < L:
            tw = target_words[i]
            if tw in PUNCT_TOKENS:
                masks[i, punct_ids] = True
                if tw in vocab:
                    masks[i, vocab[tw]] = True
            else:
                if mask_mode == "lax_vocab":
                    row = np.ones(V, dtype=bool)
                    row[special_ids] = False
                    masks[i] = row
                else:  # lcvr_full
                    masks[i] = char_mask
        elif i == L:
            masks[i, end_id] = True
        else:
            masks[i, end_id] = True
            masks[i, pad_id] = True
    return th.tensor(masks, dtype=th.bool)


def decode_with_lcvr(model, latents, position_mask, tokenizer, device):
    """latents: (B, T, d) ndarray
    position_mask: (T, V) bool tensor on cpu
    returns: list of B 个空格分隔字符串"""
    x_t = th.tensor(latents, dtype=th.float32, device=device)
    with th.no_grad():
        logits = model.get_logits(x_t)  # (B, T, V_full)
    B, T, V_full = logits.shape

    # mask: shape 与 logits 对齐（可能 V_full = V_vocab + 1，我们 pad）
    pmask = position_mask.to(device)
    if pmask.size(1) < V_full:
        pad_mask = th.zeros((T, V_full - pmask.size(1)), dtype=th.bool, device=device)
        pmask = th.cat([pmask, pad_mask], dim=1)
    elif pmask.size(1) > V_full:
        pmask = pmask[:, :V_full]
    pmask = pmask.unsqueeze(0).expand(B, -1, -1)  # (B, T, V_full)

    # Apply mask: 不允许的位置 logit = -inf
    logits = logits.masked_fill(~pmask, float("-inf"))
    ids = logits.argmax(dim=-1)  # (B, T)

    out = []
    for seq in ids.cpu().numpy():
        toks = [tokenizer[int(t)] for t in seq]
        out.append(" ".join(toks))
    return out


def main():
    args = parse_args()
    device = args.device if th.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    model, train_args = build_model(args.model_path, device)
    seq_len = train_args["image_size"] ** 2
    print(f"[info] image_size={train_args['image_size']}, seq_len={seq_len}")

    vocab = json.load(open(os.path.join(os.path.dirname(args.model_path), "vocab.json")))
    tokenizer = load_tokenizer("e2e-tgt", "random", os.path.dirname(args.model_path))
    print(f"[info] vocab size = {len(vocab)}, tokenizer size = {len(tokenizer)}")

    freq = None
    if args.corpus and args.restrict_freq != "none":
        print(f"[info] loading corpus freq from {args.corpus}")
        freq = load_freq(args.corpus, vocab)
        print(f"[info] freq quantiles: 20%={np.percentile(freq[freq>0], 20):.0f}, "
              f"80%={np.percentile(freq[freq>0], 80):.0f}")

    print(f"[info] loading targets from {args.target_json}")
    targets = []
    with open(args.target_json) as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    print(f"[info] {len(targets)} targets")

    print(f"[info] loading sample latents from {args.pkl}")
    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    sample_dict = payload["sample_dict"]
    print(f"[info] sample_dict has {len(sample_dict)} entries")

    if len(sample_dict) != len(targets):
        print(f"[warn] targets ({len(targets)}) != sample_dict entries ({len(sample_dict)}); "
              f"假设按出现顺序 1-1 对应（infill.py 是顺序处理的）")

    # 顺序匹配（infill.py 是顺序处理 control_label_lst → sample_dict）
    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    n = min(len(targets), len(sample_dict))
    sample_keys = list(sample_dict.keys())
    decoded_pairs = []
    for i in range(n):
        target = targets[i]
        latents = sample_dict[sample_keys[i]]  # (B, T, d)
        T = latents.shape[1]
        pmask = build_position_masks(target["words_"], T, vocab, freq,
                                     args.restrict_freq,
                                     mask_mode=args.mask_mode)
        words = decode_with_lcvr(model, latents, pmask, tokenizer, device)
        decoded_pairs.append((sample_keys[i], words))
        L = len(target["words_"])
        print(f"  [{i+1}/{n}] L={L}  cand0_head={words[0].split(' ', L+1)[:L][:5]}")

    print(f"\n[info] writing to {args.out}")
    with open(args.out, "w", encoding="utf-8") as f:
        for k, v in decoded_pairs:
            print({k: v}, file=f)

    print(f"[done] {len(decoded_pairs)} entries written")


if __name__ == "__main__":
    main()
