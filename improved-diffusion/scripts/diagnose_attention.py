"""Attention 诊断脚本：验证 / 排除 backbone 容量假设。

设计目标：
  1. 拿到 12-layer BERT-base encoder 在不同 timestep / 不同输入下的 attention 矩阵
  2. 计算「按距离 binning 的 attention 强度」：层 × 头 × 距离桶 → 平均权重
       - 如果远距离 (|i-j|>50) 注意力几乎为 0 → backbone 在 144 token 内都没建立
         long-range，那 Mamba/RoPE 等结构改造就有意义
       - 如果有合理的远程注意力 → backbone 容量足够，问题不在 backbone
  3. 对比 head 区 [0, L) vs tail 区 [L, 144) 的 attention pattern：
       - 如果 tail 区互相 attention 高，head 区关注 tail → tail 区在"分散注意力"
  4. 计算每层每头的 attention entropy（uniform vs spike 两端检测）

输入：
  - 训练好的 sz12 模型 (.pt)
  - 一些 ci 真值序列（用 train data 或 control_gen 的 word_lst）
  - 多个 timestep（t=0/50/100/150/199）

输出：
  - control_gen/diag_attention/heatmap_layer{i}.png  (12 张层级 attention map)
  - control_gen/diag_attention/distance_decay.png   (按 |i-j| 距离的 attention)
  - control_gen/diag_attention/head_vs_tail.txt     (区域间 attention 对比)
  - control_gen/diag_attention/_summary.md          (诊断结论)

用法：
  cd improved-diffusion
  python scripts/diagnose_attention.py \
      --model_path diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/ema_0.9999_400000.pt \
      --num_samples 20 \
      --out_dir control_gen/diag_attention
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from improved_diffusion.rounding import load_tokenizer


TIMESTEPS = [0, 50, 100, 150, 199]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--out_dir", default="control_gen/diag_attention")
    p.add_argument("--target_json", default="control_gen/target_ci_subset10.json")
    p.add_argument("--num_samples", type=int, default=20,
                   help="用多少条真实词（从 target_*.json 取）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--target_field", default="auto",
                   help="target 中 token list 的字段：auto 自动从 words_/tokens/target_toks 中找")
    p.add_argument("--seq_len", type=int, default=0,
                   help="序列长度；0 = 自动 image_size**2")
    return p.parse_args()


SEQ_LEN = 144  # default; overridden in main() once the model is loaded


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


def hack_model_to_emit_attentions(model):
    """让 BertEncoder 在 forward 时 output_attentions=True，并把结果存到 model.last_attentions。

    实现：monkey-patch model.input_transformers.forward。"""
    encoder = model.input_transformers
    orig_forward = encoder.forward

    def new_forward(*args, **kwargs):
        kwargs["output_attentions"] = True
        out = orig_forward(*args, **kwargs)
        # out is BaseModelOutputWithPastAndCrossAttentions; attentions is tuple of (B, H, T, T)
        model._last_attentions = out.attentions
        return out
    encoder.forward = new_forward
    return model


def _extract_target_tokens(target, field):
    if field == "auto":
        for f in ("words_", "tokens", "target_toks"):
            if f in target and isinstance(target[f], list):
                return target[f]
        raise KeyError(
            f"no recognised token field in target keys={list(target)}; "
            f"pass --target_field explicitly"
        )
    return target[field]


def build_x_from_target(target, model, device, target_field="auto"):
    """把一条 target token list 转成 latent x_0 = embedding(tokens) + tail freeze。"""
    vocab = json.load(open(os.path.join(os.path.dirname(model_path_global),
                                          "vocab.json")))
    toks = _extract_target_tokens(target, target_field)
    L = len(toks)
    end_id = vocab.get("END", 1)
    pad_id = vocab.get("PAD", end_id)
    ids = [vocab.get(w, vocab.get("UNK", 2)) for w in toks]
    fill_id = pad_id  # tail filler; sonnet/E2E use PAD, ci-block originally END
    ids = ids + [fill_id] * (SEQ_LEN - L)
    ids = th.tensor(ids[:SEQ_LEN], dtype=th.long, device=device).unsqueeze(0)
    with th.no_grad():
        x0 = model.word_embedding(ids)  # (1, T, in_channel)
    return x0, L


def add_noise(x0, t, diffusion, device):
    """diffusion forward q(x_t | x_0)."""
    t_tensor = th.tensor([t], device=device)
    noise = th.randn_like(x0)
    x_t = diffusion.q_sample(x0, t_tensor, noise=noise)
    return x_t, t_tensor


def collect_attentions(model, diffusion, target, device, target_field="auto"):
    """对一条 target，每个 timestep 跑一次 forward，收集 12 层 attention。

    返回：dict[t] = {"L": L, "attns": np.ndarray (num_layers, num_heads, T, T)}"""
    out = {}
    for t in TIMESTEPS:
        x0, L = build_x_from_target(target, model, device, target_field)
        x_t, t_tensor = add_noise(x0, t, diffusion, device)
        with th.no_grad():
            _ = model(x_t, t_tensor)
        attns = model._last_attentions  # tuple of (B, H, T, T)
        attn_arr = th.stack(attns, dim=0).squeeze(1).cpu().numpy()  # (num_layers, H, T, T)
        out[t] = {"L": L, "attns": attn_arr}
    return out


def metric_distance_decay(attn_arr, L, max_dist=None):
    """对 (num_layers, num_heads, T, T) 计算每个 |i-j| 的平均 attention。

    返回 dict[layer_idx] -> np.ndarray(max_dist+1) 平均值。"""
    num_layers, num_heads, T, _ = attn_arr.shape
    max_dist = max_dist or T - 1
    layer_decay = np.zeros((num_layers, max_dist + 1))
    counts = np.zeros((num_layers, max_dist + 1))
    for l in range(num_layers):
        for h in range(num_heads):
            for i in range(T):
                for j in range(T):
                    d = abs(i - j)
                    if d <= max_dist:
                        layer_decay[l, d] += attn_arr[l, h, i, j]
                        counts[l, d] += 1
    layer_decay /= np.clip(counts, 1, None)
    return layer_decay  # (num_layers, max_dist+1)


def metric_distance_decay_fast(attn_arr, max_dist=None):
    """更快的实现：用 numpy 向量化按距离 bin。"""
    num_layers, num_heads, T, _ = attn_arr.shape
    max_dist = max_dist or T - 1
    # 距离矩阵 |i-j|
    ii, jj = np.meshgrid(np.arange(T), np.arange(T), indexing="ij")
    dist = np.abs(ii - jj)  # (T, T)
    decay = np.zeros((num_layers, max_dist + 1))
    counts_per_d = np.bincount(dist.flatten(), minlength=max_dist + 1)
    for l in range(num_layers):
        layer_avg = attn_arr[l].mean(axis=0)  # average over heads -> (T, T)
        sums = np.bincount(dist.flatten(), weights=layer_avg.flatten(),
                           minlength=max_dist + 1)
        decay[l] = sums / np.clip(counts_per_d, 1, None)
    return decay


def metric_region_attention(attn_arr, L, T=144):
    """计算 head [0,L) ↔ tail [L,T) 的注意力流量。

    返回：dict 包含
      - h2h: head 区指向 head 区的总注意力（占比）
      - h2t: head 指向 tail
      - t2h: tail 指向 head
      - t2t: tail 指向 tail
    """
    num_layers, num_heads, _, _ = attn_arr.shape
    avg = attn_arr.mean(axis=(0, 1))  # (T, T) average over layer & head
    h_idx = np.arange(0, L)
    t_idx = np.arange(L, T)
    h2h = avg[np.ix_(h_idx, h_idx)].sum() / num_heads
    h2t = avg[np.ix_(h_idx, t_idx)].sum() / num_heads
    t2h = avg[np.ix_(t_idx, h_idx)].sum() / num_heads
    t2t = avg[np.ix_(t_idx, t_idx)].sum() / num_heads
    # 每个 query 位置出去的注意力总和应为 1，所以行和归一化后我们看 fraction
    h_total = h2h + h2t
    t_total = t2h + t2t
    return {
        "h2h_frac": float(h2h / max(h_total, 1e-9)),
        "h2t_frac": float(h2t / max(h_total, 1e-9)),
        "t2h_frac": float(t2h / max(t_total, 1e-9)),
        "t2t_frac": float(t2t / max(t_total, 1e-9)),
        "L": int(L),
        "T": int(T),
    }


def metric_attention_entropy(attn_arr):
    """每层平均 attention entropy。低熵 = spike，高熵 = uniform。

    BERT-base 12 头，T=144，最大熵 = log(144) ≈ 4.97。
    """
    eps = 1e-9
    num_layers = attn_arr.shape[0]
    out = []
    for l in range(num_layers):
        # (H, T, T) -> entropy per query, averaged
        a = attn_arr[l]
        ent = -(a * np.log(a + eps)).sum(-1)  # (H, T)
        out.append(float(ent.mean()))
    return out


def plot_attention_maps(attn_arr_avg_over_t, L, out_dir, layers_to_plot=(0, 3, 5, 7, 9, 11)):
    """attn_arr_avg_over_t: (num_layers, num_heads, T, T) — 跨 t/sample 平均后的版本"""
    num_layers, num_heads, T, _ = attn_arr_avg_over_t.shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, l in enumerate(layers_to_plot):
        if i >= len(axes):
            break
        avg = attn_arr_avg_over_t[l].mean(axis=0)  # (T, T)
        im = axes[i].imshow(np.log10(avg + 1e-6), cmap="viridis", aspect="auto")
        axes[i].set_title(f"Layer {l} log10(attention)")
        axes[i].axhline(L, color="red", linestyle="--", linewidth=0.8)
        axes[i].axvline(L, color="red", linestyle="--", linewidth=0.8)
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_heatmaps.png", dpi=120)
    plt.close()


def plot_distance_decay(decay, out_dir, title_suffix=""):
    """decay: (num_layers, max_dist+1)"""
    num_layers = decay.shape[0]
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.viridis
    for l in range(num_layers):
        plt.plot(decay[l], color=cmap(l / num_layers), label=f"L{l}", alpha=0.8, lw=1)
    plt.xlabel("|i - j|  (token distance)")
    plt.ylabel("mean attention weight")
    plt.title(f"Attention decay over distance {title_suffix}")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir / f"distance_decay{title_suffix}.png", dpi=120)
    plt.close()


# global for build_x_from_target 复用
model_path_global: str = ""


def main():
    global model_path_global, SEQ_LEN
    args = parse_args()
    model_path_global = args.model_path

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if th.cuda.is_available() else "cpu"
    print(f"[info] device={device}, model={args.model_path}")

    model, diffusion, train_args = load_model(args.model_path, device)
    inferred_T = int(train_args["image_size"]) ** 2
    SEQ_LEN = args.seq_len if args.seq_len > 0 else inferred_T
    print(f"[info] image_size={train_args['image_size']} → seq_len={SEQ_LEN}")

    # 抓 BERT 配置
    enc = model.input_transformers
    cfg = enc.layer[0].attention.self
    print(f"[info] BERT encoder: {len(enc.layer)} layers × "
          f"{cfg.num_attention_heads} heads × {cfg.attention_head_size} head_dim")

    hack_model_to_emit_attentions(model)

    # 加载 targets
    targets = []
    with open(args.target_json) as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    targets = targets[: args.num_samples]
    print(f"[info] using {len(targets)} targets")

    # 收集所有样本所有 timestep 的 attention
    accum_attns = defaultdict(list)  # key=t, value=list of (L, attn_arr)
    region_metrics = defaultdict(list)  # key=t, value=list of region-fraction dicts
    for i, tgt in enumerate(targets):
        out = collect_attentions(model, diffusion, tgt, device, args.target_field)
        for t, payload in out.items():
            accum_attns[t].append(payload)
            region_metrics[t].append(metric_region_attention(payload["attns"], payload["L"]))
        print(f"  [{i+1}/{len(targets)}] L={out[0]['L']}")

    # 跨 sample 平均（每 t 平均出一个 (num_layers, H, T, T)）
    print("[info] computing distance decay per t ...")
    decay_per_t = {}
    for t, payloads in accum_attns.items():
        avg_attn = np.mean([p["attns"] for p in payloads], axis=0)  # (L, H, T, T)
        decay_per_t[t] = metric_distance_decay_fast(avg_attn)

    # 选个 timestep 画 heatmap（中间步 t=100 最有代表性）
    target_t = 100 if 100 in accum_attns else TIMESTEPS[0]
    avg_attn_t = np.mean([p["attns"] for p in accum_attns[target_t]], axis=0)
    L_first = accum_attns[target_t][0]["L"]
    plot_attention_maps(avg_attn_t, L_first, out_dir)

    # decay plot（每个 t 一条曲线）
    plt.figure(figsize=(10, 6))
    cmap = plt.cm.coolwarm
    for ti, (t, decay) in enumerate(sorted(decay_per_t.items())):
        # 跨层平均
        avg = decay.mean(axis=0)
        plt.plot(avg, color=cmap(ti / len(decay_per_t)), label=f"t={t}", lw=1.5)
    plt.xlabel("|i - j|  (token distance)")
    plt.ylabel("mean attention weight (avg over layers/heads)")
    plt.title("Attention decay vs distance, by diffusion timestep")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "decay_by_timestep.png", dpi=120)
    plt.close()

    # 每层 decay (用 t=100 平均)
    plot_distance_decay(decay_per_t[target_t], out_dir, title_suffix=f"_layer_t{target_t}")

    # entropy
    entropies = {}
    for t, payloads in accum_attns.items():
        avg_attn = np.mean([p["attns"] for p in payloads], axis=0)
        entropies[t] = metric_attention_entropy(avg_attn)
    print("\n=== per-layer attention entropy (max possible = log144 ≈ 4.97) ===")
    for t in sorted(entropies.keys()):
        ent_str = ", ".join(f"L{i}={e:.2f}" for i, e in enumerate(entropies[t]))
        print(f"  t={t:3d}: {ent_str}")

    # region metrics
    region_summary = {}
    for t, ms in region_metrics.items():
        # 每个 sample 一组分数，求平均
        keys = ["h2h_frac", "h2t_frac", "t2h_frac", "t2t_frac"]
        avg = {k: float(np.mean([m[k] for m in ms])) for k in keys}
        region_summary[t] = avg
    print("\n=== region attention fraction (head=[0,L), tail=[L,T)) ===")
    print(f"  {'t':>3s}  {'h2h':>6s}  {'h2t':>6s}  {'t2h':>6s}  {'t2t':>6s}")
    for t in sorted(region_summary.keys()):
        s = region_summary[t]
        print(f"  {t:>3d}  {s['h2h_frac']:>6.1%}  {s['h2t_frac']:>6.1%}  "
              f"{s['t2h_frac']:>6.1%}  {s['t2t_frac']:>6.1%}")

    # === 写诊断结论 markdown ===
    summary = ["# Attention 诊断报告", ""]
    summary.append(f"模型：`{args.model_path}`")
    summary.append(f"BERT encoder：{len(enc.layer)} layers × {cfg.num_attention_heads} heads "
                   f"× hidden={cfg.num_attention_heads * cfg.attention_head_size}")
    summary.append(f"序列长度：{SEQ_LEN}")
    summary.append(f"评估样本：{len(targets)} 条 ci，timestep ∈ {TIMESTEPS}")
    summary.append("")
    summary.append("## 1. 距离衰减（log scale）")
    summary.append("")
    summary.append("- 文件：`distance_decay_layer_t100.png`、`decay_by_timestep.png`")
    summary.append("- 关键问：**距离 ≥ 50 的 attention 是否塌成均匀 / 噪声？** 是 → backbone 无 long-range；否 → 有")
    summary.append("")

    # 用 t=100 算"远距离 vs 近距离"比值
    avg_decay_t100 = decay_per_t[target_t].mean(axis=0)  # (max_dist+1,)
    near = avg_decay_t100[1:11].mean()       # 距离 1-10
    mid_hi = min(50, len(avg_decay_t100))
    mid = avg_decay_t100[20:mid_hi].mean() if mid_hi > 20 else 0.0
    far_lo = min(60, len(avg_decay_t100) - 1)
    far_hi = min(140, len(avg_decay_t100))
    far = avg_decay_t100[far_lo:far_hi].mean() if far_hi > far_lo else 0.0
    uniform = 1.0 / SEQ_LEN
    summary.append(f"| 距离 bin | 平均 attention | vs uniform (1/{SEQ_LEN} ≈ {uniform:.4f}) |")
    summary.append("|---|---|---|")
    summary.append(f"| 近 (1–10) | {near:.4f} | {near/uniform:.2f}× |")
    summary.append(f"| 中 (20–50) | {mid:.4f} | {mid/uniform:.2f}× |")
    summary.append(f"| 远 (60–140) | {far:.4f} | {far/uniform:.2f}× |")
    summary.append("")
    if far / uniform > 0.5 and far / uniform < 2.0:
        summary.append("**判断：远距离 attention ≈ uniform → 远程结构信号弱，但 144 还是局部主导，不算崩塌**")
    elif far / uniform < 0.3:
        summary.append("**判断：远距离 attention ≪ uniform → backbone 在 144 token 内完全没建立远程依赖（异常，可能 RoPE/ALiBi 有用）**")
    elif far / uniform > 2.0:
        summary.append("**判断：远距离 attention 显著高于 uniform → 模型有 long-range 信号，backbone 不是瓶颈**")
    summary.append("")

    summary.append("## 2. 区域注意力流量")
    summary.append("")
    summary.append("head=[0,L) 是有效内容区，tail=[L,144) 是 boundary_freeze 应该截断的区。")
    summary.append("如果 h2t（head→tail）很高，说明模型把注意力浪费在 tail 区。")
    summary.append("")
    summary.append("| t | h2h | h2t | t2h | t2t |")
    summary.append("|---|---|---|---|---|")
    for t in sorted(region_summary.keys()):
        s = region_summary[t]
        summary.append(f"| {t} | {s['h2h_frac']:.1%} | {s['h2t_frac']:.1%} | "
                       f"{s['t2h_frac']:.1%} | {s['t2t_frac']:.1%} |")
    summary.append("")
    h2t_t100 = region_summary[target_t]["h2t_frac"]
    if h2t_t100 > 0.4:
        summary.append(f"**判断：head→tail 占 {h2t_t100:.1%} > 40% → 模型大量 attention 跑去看 tail 区，导致 head 区生成被 tail 干扰；"
                       "boundary_freeze 在推理时已部分缓解，但训练时 attention 已建立这种 pattern**")
    else:
        summary.append(f"head→tail 占 {h2t_t100:.1%}，正常范围。")
    summary.append("")

    summary.append("## 3. 注意力熵")
    summary.append("")
    summary.append(f"max possible entropy = log(144) ≈ 4.97. 太低 → spike，太高 → uniform。")
    summary.append("")
    for t in sorted(entropies.keys()):
        avg_ent = np.mean(entropies[t])
        summary.append(f"- t={t:3d}: 跨层平均 = {avg_ent:.2f}")
    summary.append("")

    summary.append("## 4. 结论")
    summary.append("")

    # 自动判断
    far_ratio = far / uniform
    if far_ratio < 0.3:
        verdict = "**backbone 容量是瓶颈**：远距离 attention 已塌缩，建议先试 RoPE / ALiBi 再考虑 SSM。"
    elif h2t_t100 > 0.4:
        verdict = ("**backbone 容量充足，但训练目标错配**：远距离 attention 健康，但 head 注意力被 tail 拉走 "
                   "→ 真正的 fix 是 length-conditional / boundary-aware retraining，而非换 SSM。")
    else:
        verdict = "**backbone 容量充足且 attention pattern 健康**：换 SSM 大概率不会带来增益，问题在 vocab projection / control / 数据。"
    summary.append(verdict)

    (out_dir / "_summary.md").write_text("\n".join(summary), encoding="utf-8")

    # 也保存原始 numpy 备查
    np.savez_compressed(out_dir / "attentions.npz",
                        decay_per_t=np.stack([decay_per_t[t] for t in sorted(decay_per_t.keys())]),
                        timesteps=np.array(sorted(decay_per_t.keys())))

    print(f"\n[done] 报告写入 {out_dir / '_summary.md'}")


if __name__ == "__main__":
    main()
