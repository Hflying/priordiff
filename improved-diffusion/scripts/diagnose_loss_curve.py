"""从训练 stdout log 抽取 loss 曲线，画出来 + 给出 plateau 判断。

这是 attention/vocab 之外的第三个诊断：训练容量是否打满。

输入：训练 log（含 logger 表 `| loss | ... |`、`| step | ... |`）
输出：control_gen/diag_loss/loss_curve.png + _summary.md
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAT_KV = re.compile(r"^\| (\w+)\s+\| ([\-0-9.e+]+)\s+\|")
PAT_DIVIDER = re.compile(r"^-{10,}")


def parse_log(paths: list[Path]):
    """合并多个 log 文件，提取所有 metric blocks（每个 ----- 分隔）"""
    blocks = []
    cur = {}
    inside = False
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                if PAT_DIVIDER.match(line):
                    if cur:
                        blocks.append(cur)
                        cur = {}
                    inside = not inside if cur else True
                    continue
                m = PAT_KV.match(line)
                if m:
                    k, v = m.group(1), m.group(2)
                    try:
                        cur[k] = float(v)
                    except ValueError:
                        pass
        if cur:
            blocks.append(cur)
            cur = {}
    return blocks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", nargs="+", required=True, help="训练 stdout log 文件，可多个，按时间顺序")
    p.add_argument("--out_dir", default="control_gen/diag_loss")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocks = parse_log([Path(p) for p in args.logs])
    print(f"[info] parsed {len(blocks)} metric blocks")

    # 分两类：train metric block (有 loss/mse/step) vs eval block (有 eval_loss)
    train_blocks = [b for b in blocks if "step" in b and "loss" in b and "eval_loss" not in b]
    eval_blocks  = [b for b in blocks if "eval_loss" in b]
    print(f"[info] train blocks: {len(train_blocks)}, eval blocks: {len(eval_blocks)}")

    # 训练曲线
    steps = np.array([b["step"] for b in train_blocks])
    loss = np.array([b.get("loss", np.nan) for b in train_blocks])
    mse = np.array([b.get("mse", np.nan) for b in train_blocks])
    mse_q0 = np.array([b.get("mse_q0", np.nan) for b in train_blocks])
    mse_q1 = np.array([b.get("mse_q1", np.nan) for b in train_blocks])
    mse_q2 = np.array([b.get("mse_q2", np.nan) for b in train_blocks])
    mse_q3 = np.array([b.get("mse_q3", np.nan) for b in train_blocks])

    # 排序（防止重复 / 乱序）
    order = np.argsort(steps)
    steps = steps[order]; loss = loss[order]; mse = mse[order]
    mse_q0 = mse_q0[order]; mse_q1 = mse_q1[order]; mse_q2 = mse_q2[order]; mse_q3 = mse_q3[order]

    print(f"\n=== training loss summary ===")
    print(f"  steps range: {steps.min():.0f} → {steps.max():.0f}")
    print(f"  blocks: {len(steps)}")
    print(f"\n  loss:    {loss[0]:.4f} (start) → {loss[-1]:.4f} (end), Δ = {loss[-1]-loss[0]:+.4f}")
    print(f"  mse:     {mse[0]:.4f} → {mse[-1]:.4f}, Δ = {mse[-1]-mse[0]:+.4f}")
    print(f"  mse_q0:  {mse_q0[0]:.4f} → {mse_q0[-1]:.4f}, Δ = {mse_q0[-1]-mse_q0[0]:+.4f}  (low noise t)")
    print(f"  mse_q1:  {mse_q1[0]:.4f} → {mse_q1[-1]:.4f}, Δ = {mse_q1[-1]-mse_q1[0]:+.4f}")
    print(f"  mse_q2:  {mse_q2[0]:.4f} → {mse_q2[-1]:.4f}, Δ = {mse_q2[-1]-mse_q2[0]:+.4f}")
    print(f"  mse_q3:  {mse_q3[0]:.4f} → {mse_q3[-1]:.4f}, Δ = {mse_q3[-1]-mse_q3[0]:+.4f}  (high noise t)")

    # plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(steps, loss, label="loss", color="C0", alpha=0.7)
    axes[0].plot(steps, mse, label="mse", color="C1", alpha=0.7)
    axes[0].set_ylabel("loss / mse")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title("Training loss curve")

    axes[1].plot(steps, mse_q0, label="mse_q0 (low noise t)", color="C0", alpha=0.7)
    axes[1].plot(steps, mse_q1, label="mse_q1", color="C2", alpha=0.7)
    axes[1].plot(steps, mse_q2, label="mse_q2", color="C3", alpha=0.7)
    axes[1].plot(steps, mse_q3, label="mse_q3 (high noise t)", color="C1", alpha=0.7)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("mse by t-quantile")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png", dpi=120)
    plt.close()

    # plateau check：最后 25% 步 vs 中间 25% 步的均值
    n = len(steps)
    quart = n // 4
    mid_loss = loss[quart * 1: quart * 2].mean()
    end_loss = loss[-quart:].mean()
    mid_mse_q0 = mse_q0[quart: quart * 2].mean()
    end_mse_q0 = mse_q0[-quart:].mean()

    md = ["# 训练曲线诊断 (loss plateau check)", ""]
    md.append(f"日志：{', '.join(args.logs)}")
    md.append(f"训练 step 范围：{steps.min():.0f} → {steps.max():.0f}")
    md.append(f"metric blocks: {len(steps)}")
    md.append("")
    md.append("## 1. Loss 演化")
    md.append("")
    md.append("| metric | start | end | Δ |")
    md.append("|---|---|---|---|")
    md.append(f"| loss    | {loss[0]:.4f} | {loss[-1]:.4f} | {loss[-1]-loss[0]:+.4f} |")
    md.append(f"| mse     | {mse[0]:.4f} | {mse[-1]:.4f} | {mse[-1]-mse[0]:+.4f} |")
    md.append(f"| mse_q0  | {mse_q0[0]:.4f} | {mse_q0[-1]:.4f} | {mse_q0[-1]-mse_q0[0]:+.4f} |")
    md.append(f"| mse_q1  | {mse_q1[0]:.4f} | {mse_q1[-1]:.4f} | {mse_q1[-1]-mse_q1[0]:+.4f} |")
    md.append(f"| mse_q2  | {mse_q2[0]:.4f} | {mse_q2[-1]:.4f} | {mse_q2[-1]-mse_q2[0]:+.4f} |")
    md.append(f"| mse_q3  | {mse_q3[0]:.4f} | {mse_q3[-1]:.4f} | {mse_q3[-1]-mse_q3[0]:+.4f} |")
    md.append("")
    md.append("## 2. Plateau 判断（中段 vs 末段均值）")
    md.append("")
    md.append("| metric | 中段（第 2/4 段）| 末段（第 4/4 段）| 相对变化 |")
    md.append("|---|---|---|---|")
    rel_loss = (end_loss - mid_loss) / mid_loss * 100
    rel_q0   = (end_mse_q0 - mid_mse_q0) / mid_mse_q0 * 100
    md.append(f"| loss    | {mid_loss:.4f} | {end_loss:.4f} | {rel_loss:+.2f}% |")
    md.append(f"| mse_q0  | {mid_mse_q0:.4f} | {end_mse_q0:.4f} | {rel_q0:+.2f}% |")
    md.append("")

    md.append("## 3. 结论")
    md.append("")
    if abs(rel_loss) < 2:
        md.append(f"**Loss 已 plateau**：末段相对中段变化 {rel_loss:+.2f}%（< 2%），说明训练量已撑到模型容量。"
                  f"再训也没用，要换 architecture / 改目标函数才有突破。")
    elif rel_loss < -5:
        md.append(f"**Loss 仍在下降**：末段相对中段下降 {-rel_loss:.2f}%，"
                  f"说明 backbone 容量没用满，纯加训练步数仍可改善。")
    else:
        md.append(f"**Loss 接近 plateau**：末段相对中段变化 {rel_loss:+.2f}%（介于 2-5%），"
                  f"训练量边际收益变小，但还没完全打满。LR-anneal 已到末段，再延 LR-warm restart 可能能再压一段。")
    md.append("")
    md.append("如要『换 backbone 解决 capacity 问题』，**前提是 loss 已 plateau**。本日志结果可作判据。")

    (out_dir / "_summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n[done] 写入 {out_dir / '_summary.md'}")


if __name__ == "__main__":
    main()
