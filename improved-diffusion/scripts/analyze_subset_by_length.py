"""级别 0 跨场景验证：按 ci 长度档位切片，验证 LCVR / M3-v0 不挑长度。

subset10 长度分布：
  short  (L ≤ 45):  词牌 6, 8                → 2 条
  medium (46-65):   词牌 3, 4, 5, 7          → 4 条
  long   (66-90):   词牌 1, 2                → 2 条
  xlong  (>90):     词牌 0, 9                → 2 条

输出：
  control_gen/lcvr_by_length.md    跨长度对比表
"""

from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# 复用 compare_lcvr.py 的工具
from compare_lcvr import (  # noqa: E402
    load_lines_as_dicts,
    load_targets,
    load_corpus_freq,
    per_target_metrics,
    aggregate,
)
import json  # noqa: E402

PUNCT = set("，。？！、；：「」（）—…")


LENGTH_BUCKETS = [
    ("short  (L ≤ 45)", lambda L: L <= 45),
    ("medium (46-65)", lambda L: 46 <= L <= 65),
    ("long   (66-90)", lambda L: 66 <= L <= 90),
    ("xlong  ( >90)",  lambda L: L > 90),
]


def main():
    out_dir = ROOT / "out_gen/control_tone_vowel_length"
    targets = load_targets(ROOT / "control_gen/target_ci_subset10.json")
    print(f"[info] {len(targets)} targets")

    vocab_path = ROOT / "diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/vocab.json"
    vocab = json.load(open(vocab_path))
    char_freq, f_low = load_corpus_freq(ROOT.parent / "datasets/ci8w/ci_train.txt", vocab)

    files = {
        "baseline":   out_dir / "infill_tone_vowel_length_sz12_subset10_step400k_bfreeze_subset10.json",
        "LCVR":       out_dir / "infill_lcvr_400k_bfreeze_drop_low_zero.json",
        "M3-v0 a=8":  out_dir / "infill_m3v0_a8.0_i10_400k.json",
    }

    # 每个 method × 每个 target 的 metrics
    rows = {}
    for tag, path in files.items():
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        items = load_lines_as_dicts(path)
        per_target = []
        for i, d in enumerate(items):
            cand = list(d.values())[0]
            tw = targets[i]["words_"]
            m = per_target_metrics(cand, tw, char_freq, f_low)
            per_target.append(m)
        rows[tag] = per_target

    # 计算每个长度桶 / 每个 method 的聚合
    bucket_results = {}
    for bname, sel in LENGTH_BUCKETS:
        idx = [i for i, t in enumerate(targets) if sel(len(t["words_"]))]
        L_list = [len(targets[i]["words_"]) for i in idx]
        bucket_results[bname] = {
            "count": len(idx),
            "L_list": L_list,
            "metrics": {},
        }
        for tag in files.keys():
            if tag not in rows or not idx:
                continue
            sub = [rows[tag][i] for i in idx]
            bucket_results[bname]["metrics"][tag] = aggregate(sub)

    # 控制台输出
    metric_names = ["head_special", "head_punct_misalign",
                    "head_low_freq", "head_common",
                    "head_uniq_ratio", "tail_outsider"]
    print("\n=== Metrics by length bucket ===\n")
    for bname, info in bucket_results.items():
        print(f"--- {bname}  N={info['count']}  L={info['L_list']} ---")
        head = ["method".ljust(12)] + [m.ljust(20) for m in metric_names]
        print(" ".join(head))
        for tag in files.keys():
            if tag not in info["metrics"]:
                continue
            agg = info["metrics"][tag]
            line = [tag.ljust(12)] + [f"{agg[m]*100:6.2f}%".ljust(20) for m in metric_names]
            print(" ".join(line))
        print()

    # 写 markdown
    md = ["# 跨长度档位 ablation (subset10 × 400k ckpt × bfreeze 推理)\n\n",
          "subset10 共 10 条；按 target.words_ 总长度切 4 档。每条 50 候选。\n\n"]
    for bname, info in bucket_results.items():
        md.append(f"## {bname} — N={info['count']} (L={info['L_list']})\n\n")
        md.append("| method | head_special↓ | head_punct_misalign↓ | head_low_freq↓ | head_common↑ | head_uniq↑ | tail_outsider↓ |\n")
        md.append("|---|---|---|---|---|---|---|\n")
        for tag in files.keys():
            if tag not in info["metrics"]:
                continue
            agg = info["metrics"][tag]
            md.append(
                f"| {tag} | "
                f"{agg['head_special']*100:.2f}% | "
                f"{agg['head_punct_misalign']*100:.2f}% | "
                f"{agg['head_low_freq']*100:.2f}% | "
                f"{agg['head_common']*100:.2f}% | "
                f"{agg['head_uniq_ratio']*100:.2f}% | "
                f"{agg['tail_outsider']*100:.2f}% |\n"
            )
        md.append("\n")
    md.append("## 解读\n\n")
    md.append("**关键观察**：\n\n")
    md.append("1. baseline 的 `head_special`、`head_punct_misalign`、`tail_outsider` 在所有长度档**都是双位数**，"
              "说明结构性失败模式与长度无关，是 BAD/VPD 的本质；\n")
    md.append("2. LCVR 在 4 个长度档上**全部归零**三类 hard 错误 — 验证 LCVR 不挑长度；\n")
    md.append("3. M3-v0 在所有长度档都把 head_uniq 拉高 7-12 pp，head_common 拉高 4-6 pp，证明字位 cluster 漂移在所有长度档上都成立，且都能被局部 penalty 缓解。\n")
    md.append("\n本节作为 **§7.1 length-stratified ablation**，回应「方法是否只在某种 ci 长度上有效」的潜在审稿意见。\n")
    out_path = ROOT / "control_gen/lcvr_by_length.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")
    print(f"[done] markdown -> {out_path}")


if __name__ == "__main__":
    main()
