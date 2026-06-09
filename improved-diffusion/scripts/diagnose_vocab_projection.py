"""Vocab projection 诊断脚本：定位 latent → token 投影中"低频字漂移"的几何根因。

设计目标：
  1. 统计训练语料 (ci_train.txt) 的 token 频次
  2. 加载 word_embedding 矩阵 (vocab_size, 16)，计算每个 token 的 Voronoi cell 半径
       = 离最近邻 token 的距离 → 越大表示该 cell 越大 → 该 token 越容易"被吸进来"
  3. 对比 高频 / 中频 / 低频 三档的 cell 半径分布
  4. 加载生成的 latent (.pkl)，算每个位置的 latent 到最近 vocab 的距离分布
       - 如果生成 latent 普遍掉到"低频字 cell 中心"，说明 prior 飘到稀疏区
  5. 计算"投影信任度"：argmax 投到的 token 与第 2 近 token 的距离差（margin）
       - margin 小 → 模型其实摇摆，但 argmax 强行选了一个

输出：
  - control_gen/diag_vocab/cell_radius_by_freq.png
  - control_gen/diag_vocab/latent_to_vocab_distance.png
  - control_gen/diag_vocab/projection_margin.png
  - control_gen/diag_vocab/_summary.md

用法：
  cd improved-diffusion
  python scripts/diagnose_vocab_projection.py \
      --model_path diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/ema_0.9999_400000.pt \
      --corpus     ../datasets/ci8w/ci_train.txt \
      --pkl        out_gen/control_tone_vowel_length/checkpoint_tone_vowel_length_sz12_subset10_step400k_bfreeze.pkl \
      --out_dir    control_gen/diag_vocab
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--corpus", default="../datasets/ci8w/ci_train.txt")
    p.add_argument("--pkl", default=None,
                   help="可选：生成的 sample_dict pkl，用于看真实 latent 分布")
    p.add_argument("--out_dir", default="control_gen/diag_vocab")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_corpus_freq(corpus_path: str, vocab: dict) -> Counter:
    """统计语料 token 频次（按 vocab 切字）。"""
    cnt = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            # 数据格式：title|content
            parts = line.rstrip("\n").split("|", 1)
            text = parts[1] if len(parts) >= 2 else parts[0]
            for ch in text:
                cnt[ch] += 1
    return cnt


def load_model(model_path, device):
    cfg_path = os.path.join(os.path.dirname(model_path), "training_args.json")
    with open(cfg_path) as f:
        train_args = json.load(f)
    defaults = model_and_diffusion_defaults()
    init_kwargs = {k: train_args.get(k, v) for k, v in defaults.items()}
    model, diffusion = create_model_and_diffusion(**init_kwargs)
    state = th.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, diffusion, train_args


def compute_cell_radius(embeddings: th.Tensor) -> np.ndarray:
    """对每个 token，找最近邻的距离。
    embeddings: (V, d), torch tensor on device.
    返回：np.ndarray (V,) of nearest-neighbor distances (excluding self).
    """
    V = embeddings.size(0)
    # cdist + diagonal 替换为 inf
    radii = np.zeros(V, dtype=np.float32)
    block = 256  # 分块算，避免 V*V 大矩阵
    for i in range(0, V, block):
        e = embeddings[i:i+block]  # (b, d)
        d = th.cdist(e, embeddings, p=2)  # (b, V)
        # 自己到自己的距离设为 inf
        for j in range(d.size(0)):
            d[j, i + j] = float("inf")
        nn_dist, _ = d.min(dim=1)
        radii[i:i+block] = nn_dist.cpu().numpy()
    return radii


def get_latent_to_vocab_stats(latents: np.ndarray, embeddings: np.ndarray,
                               freq_arr: np.ndarray, top_k: int = 2):
    """对每个 latent 算到最近 vocab token 的距离 + margin（最近 vs 次近）。

    latents: (N, d) 已 reshape 为单 token 的 N 个 latent
    embeddings: (V, d)
    freq_arr: (V,) 训练频次

    返回：dict
      - dist_min: (N,) 到最近 vocab 的距离
      - margin: (N,) 第 2 近 - 第 1 近
      - argmin_freq: (N,) argmin token 的训练频次
    """
    lat_t = th.tensor(latents, dtype=th.float32)
    emb_t = th.tensor(embeddings, dtype=th.float32)
    if th.cuda.is_available():
        lat_t = lat_t.cuda()
        emb_t = emb_t.cuda()

    N = lat_t.size(0)
    chunk = 4096
    dist_min = np.zeros(N, dtype=np.float32)
    margin = np.zeros(N, dtype=np.float32)
    argmin_freq = np.zeros(N, dtype=np.int64)
    for i in range(0, N, chunk):
        l = lat_t[i:i+chunk]
        d = th.cdist(l, emb_t, p=2)  # (b, V)
        topk = th.topk(d, k=top_k, dim=1, largest=False)
        nn1 = topk.values[:, 0]
        nn2 = topk.values[:, 1] if top_k >= 2 else nn1 * 0
        ids1 = topk.indices[:, 0]
        dist_min[i:i+chunk] = nn1.cpu().numpy()
        margin[i:i+chunk] = (nn2 - nn1).cpu().numpy()
        argmin_freq[i:i+chunk] = freq_arr[ids1.cpu().numpy()]
    return {"dist_min": dist_min, "margin": margin, "argmin_freq": argmin_freq}


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if th.cuda.is_available() else "cpu"
    print(f"[info] device={device}")

    model, diffusion, _ = load_model(args.model_path, device)
    vocab = json.load(open(os.path.join(os.path.dirname(args.model_path), "vocab.json")))
    inv_vocab = {v: k for k, v in vocab.items()}
    V = len(vocab)
    print(f"[info] vocab size = {V}")

    embeddings = model.word_embedding.weight.detach()  # (V_emb, d) torch tensor on cuda
    V_emb = embeddings.size(0)
    print(f"[info] embedding shape = {tuple(embeddings.shape)} (vocab.json size={V}, emb rows={V_emb})")

    print(f"[info] computing per-token cell radius (NN distance)...")
    radii = compute_cell_radius(embeddings)

    # 统计训练频次（按 emb size 对齐，多出来的 row（V → V_emb）频次为 0）
    print(f"[info] loading corpus freq from {args.corpus}")
    char_freq = load_corpus_freq(args.corpus, vocab)
    freq_arr = np.zeros(V_emb, dtype=np.int64)
    for tok, fid in vocab.items():
        if fid < V_emb:
            freq_arr[fid] = char_freq.get(tok, 0)
    V = V_emb  # 后续用 emb size 作为基准
    print(f"[info] tokens with freq>0: {(freq_arr > 0).sum()} / {V}")
    print(f"[info] freq quantiles: 25%={np.percentile(freq_arr[freq_arr>0], 25):.0f}, "
          f"50%={np.percentile(freq_arr[freq_arr>0], 50):.0f}, "
          f"75%={np.percentile(freq_arr[freq_arr>0], 75):.0f}, "
          f"99%={np.percentile(freq_arr[freq_arr>0], 99):.0f}")

    # 按频次分档
    pos_mask = freq_arr > 0
    f_high = np.percentile(freq_arr[pos_mask], 80)  # top 20% 频次
    f_low  = np.percentile(freq_arr[pos_mask], 20)  # bottom 20%

    cat = np.full(V, "mid", dtype=object)
    cat[freq_arr >= f_high] = "high"
    cat[freq_arr <= f_low]  = "low"
    cat[freq_arr == 0]      = "zero"
    print(f"[info] freq buckets: high={ (cat=='high').sum() }, "
          f"mid={ (cat=='mid').sum() }, low={ (cat=='low').sum() }, "
          f"zero={ (cat=='zero').sum() }")

    # cell radius 对比
    summary = {}
    for c in ["high", "mid", "low", "zero"]:
        m = cat == c
        if m.sum() == 0:
            continue
        summary[c] = {
            "n": int(m.sum()),
            "radius_mean": float(radii[m].mean()),
            "radius_median": float(np.median(radii[m])),
            "radius_p10": float(np.percentile(radii[m], 10)),
            "radius_p90": float(np.percentile(radii[m], 90)),
        }
    print(f"\n=== cell radius (nearest-neighbor distance) by freq bucket ===")
    print(f"  {'bucket':>6s}  {'n':>5s}  {'mean':>8s}  {'median':>8s}  {'p10':>8s}  {'p90':>8s}")
    for c, s in summary.items():
        print(f"  {c:>6s}  {s['n']:>5d}  {s['radius_mean']:>8.3f}  "
              f"{s['radius_median']:>8.3f}  {s['radius_p10']:>8.3f}  {s['radius_p90']:>8.3f}")

    # plot
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, np.percentile(radii, 99), 60)
    for c, color in zip(["high", "mid", "low", "zero"], ["C0", "C1", "C2", "C3"]):
        m = cat == c
        if m.sum() == 0:
            continue
        plt.hist(radii[m], bins=bins, alpha=0.5, label=f"{c} (n={m.sum()})",
                 color=color, density=True)
    plt.xlabel("nearest-neighbor distance (≈ Voronoi cell radius lower bound)")
    plt.ylabel("density")
    plt.title("Per-token cell radius vs training freq bucket")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cell_radius_by_freq.png", dpi=120)
    plt.close()

    # === 实际生成 latent 的诊断 ===
    latent_diag = None
    if args.pkl:
        print(f"\n[info] loading sample latents from {args.pkl}")
        with open(args.pkl, "rb") as f:
            payload = pickle.load(f)
        sd = payload["sample_dict"]
        # 把所有 (B, T, d) 拼成 (N, d)
        lats = []
        for k, arr in sd.items():
            lats.append(arr.reshape(-1, arr.shape[-1]))
        all_lat = np.concatenate(lats, axis=0)
        print(f"[info] total latent points: {all_lat.shape}")

        # 抽样 50k 防止 OOM
        if all_lat.shape[0] > 50000:
            idx = np.random.choice(all_lat.shape[0], 50000, replace=False)
            all_lat = all_lat[idx]

        emb_np = embeddings.cpu().numpy()
        latent_diag = get_latent_to_vocab_stats(all_lat, emb_np, freq_arr)

        # 投到 high/mid/low/zero 的占比
        argmin_cat = np.where(latent_diag["argmin_freq"] >= f_high, "high",
                     np.where(latent_diag["argmin_freq"] >= f_low,  "mid",
                     np.where(latent_diag["argmin_freq"] > 0,       "low", "zero")))
        from collections import Counter as Cnt
        cnt = Cnt(argmin_cat.tolist())
        total = len(argmin_cat)
        print(f"\n=== latent 投影目的地（按频次分档）===")
        for c in ["high", "mid", "low", "zero"]:
            print(f"  {c}: {cnt[c]} ({cnt[c]/total:.1%})")

        # 距离 / margin 分布
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.hist(latent_diag["dist_min"], bins=50)
        plt.title("latent → nearest vocab distance")
        plt.xlabel("L2 distance")
        plt.subplot(1, 2, 2)
        plt.hist(latent_diag["margin"], bins=50)
        plt.title("projection margin (2nd - 1st nearest)")
        plt.xlabel("L2 margin")
        plt.tight_layout()
        plt.savefig(out_dir / "latent_to_vocab_distance.png", dpi=120)
        plt.close()

    # === 写报告 ===
    md = ["# Vocab Projection 诊断报告", ""]
    md.append(f"模型：`{args.model_path}`")
    md.append(f"语料：`{args.corpus}`")
    md.append(f"vocab 大小：{V}")
    md.append(f"embedding 维度：{embeddings.shape[1]}")
    md.append("")
    md.append("## 1. 词频分布")
    md.append("")
    md.append(f"- 出现过的 token: {int(pos_mask.sum())} / {V}")
    md.append(f"- 高频 (top 20%): freq ≥ {int(f_high)}")
    md.append(f"- 低频 (bottom 20%): freq ≤ {int(f_low)}")
    md.append(f"- 训练频次为 0 的 token: {int((cat=='zero').sum())} 个（START/END/PAD/UNK 等）")
    md.append("")
    md.append("## 2. Voronoi cell 半径（≈ NN 距离）")
    md.append("")
    md.append("| bucket | n | mean | median | p10 | p90 |")
    md.append("|---|---|---|---|---|---|")
    for c, s in summary.items():
        md.append(f"| {c} | {s['n']} | {s['radius_mean']:.3f} | "
                  f"{s['radius_median']:.3f} | {s['radius_p10']:.3f} | "
                  f"{s['radius_p90']:.3f} |")
    md.append("")
    if "low" in summary and "high" in summary:
        ratio = summary["low"]["radius_median"] / summary["high"]["radius_median"]
        md.append(f"**低频字 / 高频字 cell 半径中位数比 = {ratio:.2f}×**")
        if ratio > 1.5:
            md.append("→ 低频字 cell 显著更大，意味着任何 latent 飘到该区域都会被吸过去。"
                      "这就是 projection gap 的几何根因。")
        else:
            md.append("→ 各档 cell 半径相近，projection 不偏向某档。")
    md.append("")

    if latent_diag is not None:
        md.append("## 3. 真实生成 latent 投影分析")
        md.append("")
        md.append(f"- 来源：`{args.pkl}`")
        md.append(f"- 采样点数：{len(latent_diag['dist_min'])}")
        md.append(f"- latent → 最近 vocab 距离 中位数: {np.median(latent_diag['dist_min']):.3f}")
        md.append(f"- 投影 margin 中位数: {np.median(latent_diag['margin']):.3f}")
        md.append(f"  （margin 越小，模型在 top-1 / top-2 间越摇摆）")
        md.append("")
        md.append("**投影目的地分档**：")
        for c in ["high", "mid", "low", "zero"]:
            md.append(f"- {c}: {cnt[c]} ({cnt[c]/total:.1%})")
        md.append("")
        # 训练分布参考
        train_frac = {}
        total_chars = freq_arr.sum()
        for c in ["high", "mid", "low", "zero"]:
            m = cat == c
            train_frac[c] = freq_arr[m].sum() / total_chars
        md.append("**对比训练数据分布**：")
        for c in ["high", "mid", "low", "zero"]:
            md.append(f"- {c}: 训练占 {train_frac[c]:.1%}")
        md.append("")
        # 关键 misalign 检查
        gen_low = cnt["low"] / total
        train_low = train_frac["low"]
        if gen_low > train_low * 2:
            md.append(f"**判断：生成 latent 投到低频字 = {gen_low:.1%}, "
                      f"训练时低频字仅占 {train_low:.1%} —— 显著漂移。**")
            md.append("这就是 vocab projection gap 的实证：模型生成的 latent 比训练分布更偏向低频字 cell。")
        else:
            md.append(f"生成投影分布 ({gen_low:.1%} low) 与训练分布 ({train_low:.1%} low) 接近，无明显漂移。")

    (out_dir / "_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] 报告写入 {out_dir / '_summary.md'}")

    # save raw
    np.savez_compressed(out_dir / "diag_data.npz",
                        cell_radii=radii,
                        freq_arr=freq_arr,
                        category=np.array([c for c in cat]))


if __name__ == "__main__":
    main()
