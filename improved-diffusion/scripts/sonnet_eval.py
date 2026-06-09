"""评测 sonnet baseline vs LCVR 输出。

输入：sonnet_sample_lcvr.py 产出的 json
  {
    "baseline": [[tok,...], ...],   # N 个候选，每条独立解码
    "lcvr": {idx_str: {"tokens": [[...]], "n_tokens": int, "src_idx": int}}
  }
target: control_gen/target_son10.json

指标定义（与 ci LCVR 同构）：
  - head_special    : word slot 中出现 _PAD/_GO/_EOS/_UNK 的比例
  - head_eos_misalign: 句中(<,eos,>) 三元组出现位置不在 target 模板上的比例 *
  - tail_outsider    : tail 区出现非 _PAD 的比例
  - head_uniq_ratio  : word slot 中独立 word 数 / word slot 总数

* 由于 baseline 解出的位置可能不构成 14 个完整三元组，
  head_eos_misalign = 1 - len(matched_triplets)/14

输出：
  - control_gen/sonnet_compare.md   表格
  - control_gen/decoded_son10/<idx>_<src>.txt  每首前 5 候选
"""

from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


SPECIAL_BASE = {"START", "END", "UNK", "PAD"}
LINE_TRIPLET = {"<", "eos", ">"}


def load_targets(path: Path):
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def metrics_for_candidate(cand_tokens, target_rec):
    """对一条候选（长 T 的 token list）和 target 模板算指标。"""
    T = len(cand_tokens)
    lt_pos = set(target_rec["lt_positions"])
    eos_pos = set(target_rec["eos_positions"])
    gt_pos = set(target_rec["gt_positions"])
    last_triplet_end = max(target_rec["gt_positions"])
    triplet_set = lt_pos | eos_pos | gt_pos

    # word slots = i ≤ last_triplet_end and i not in triplet
    word_slots = [i for i in range(last_triplet_end + 1) if i not in triplet_set]
    tail_slots = [i for i in range(last_triplet_end + 1, T)]

    # head_special
    spec_cnt = sum(1 for i in word_slots if cand_tokens[i] in SPECIAL_BASE)
    head_special = spec_cnt / max(1, len(word_slots))

    # eos misalign: count how many triplet positions actually contain expected token
    expected = {}
    for p in target_rec["lt_positions"]:
        expected[p] = "<"
    for p in target_rec["eos_positions"]:
        expected[p] = "eos"
    for p in target_rec["gt_positions"]:
        expected[p] = ">"
    correct = sum(1 for p, e in expected.items() if p < T and cand_tokens[p] == e)
    head_eos_misalign = 1.0 - correct / 42

    # tail outsider
    if tail_slots:
        out_cnt = sum(1 for i in tail_slots if cand_tokens[i] != "PAD")
        tail_outsider = out_cnt / len(tail_slots)
    else:
        tail_outsider = 0.0

    # head uniq
    word_toks = [cand_tokens[i] for i in word_slots if cand_tokens[i] not in SPECIAL_BASE]
    if word_toks:
        head_uniq = len(set(word_toks)) / len(word_toks)
    else:
        head_uniq = 0.0

    return dict(
        head_special=head_special,
        head_eos_misalign=head_eos_misalign,
        tail_outsider=tail_outsider,
        head_uniq=head_uniq,
        word_slot_count=len(word_slots),
    )


def aggregate(per_target_metrics):
    """list of dicts -> mean dict."""
    if not per_target_metrics:
        return {}
    keys = [k for k in per_target_metrics[0] if k != "word_slot_count"]
    out = {}
    for k in keys:
        out[k] = sum(d[k] for d in per_target_metrics) / len(per_target_metrics)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_json", required=True, help="sonnet_sample_lcvr 产出 json")
    p.add_argument("--target_json", default="control_gen/target_son10.json")
    p.add_argument("--out_md", default="control_gen/sonnet_compare.md")
    p.add_argument("--out_dir", default="control_gen/decoded_son10")
    args = p.parse_args()

    out_json_path = Path(args.out_json)
    if not out_json_path.is_absolute():
        out_json_path = ROOT / out_json_path
    target_json_path = Path(args.target_json)
    if not target_json_path.is_absolute():
        target_json_path = ROOT / target_json_path

    print(f"[info] reading {out_json_path}")
    with open(out_json_path, "r") as f:
        data = json.load(f)
    targets = load_targets(target_json_path)

    n_targets = len(targets)

    # 对每条 target，分别用 baseline 和 lcvr 的全部 50 候选算指标
    cfgs = ["baseline", "lcvr"]
    if "m3v1" in data:
        cfgs.append("m3v1")
    rows = {c: [] for c in cfgs}
    per_target_avg = {c: [] for c in cfgs}

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    has_baseline = bool(data.get("baseline"))
    if not has_baseline and "baseline" in cfgs:
        cfgs.remove("baseline")
        per_target_avg.pop("baseline", None)
        rows.pop("baseline", None)

    for tgt in targets:
        idx = str(tgt["subset_idx"])
        lcvr_cands = data["lcvr"][idx]["tokens"]
        m3v1_cands = data["m3v1"][idx]["tokens"] if "m3v1" in data else None

        if has_baseline:
            baseline_cands = data["baseline"]
            bm = [metrics_for_candidate(c, tgt) for c in baseline_cands]
            per_target_avg["baseline"].append({k: sum(d[k] for d in bm) / len(bm) for k in bm[0] if k != "word_slot_count"})

        lm = [metrics_for_candidate(c, tgt) for c in lcvr_cands]
        per_target_avg["lcvr"].append({k: sum(d[k] for d in lm) / len(lm) for k in lm[0] if k != "word_slot_count"})
        if m3v1_cands is not None:
            mm = [metrics_for_candidate(c, tgt) for c in m3v1_cands]
            per_target_avg["m3v1"].append({k: sum(d[k] for d in mm) / len(mm) for k in mm[0] if k != "word_slot_count"})

        # write per-target 文件
        readable_path = out_dir / f"{int(idx):02d}_son{tgt['src_idx']}.txt"
        with open(readable_path, "w", encoding="utf-8") as f:
            f.write(f"== sonnet {idx} (src={tgt['src_idx']}, L={tgt['n_tokens']}) ==\n")
            f.write("\n--- 真值 ---\n")
            for ln in tgt["lines"]:
                f.write(" ".join(ln) + "\n")
            if has_baseline:
                f.write("\n--- baseline 候选 1 ---\n")
                f.write(" ".join(baseline_cands[0]) + "\n")
            f.write("\n--- LCVR 候选 1 ---\n")
            f.write(" ".join(lcvr_cands[0]) + "\n")
            f.write("\n--- LCVR 候选 2 ---\n")
            f.write(" ".join(lcvr_cands[1] if len(lcvr_cands) > 1 else lcvr_cands[0]) + "\n")
            f.write("\n--- LCVR 候选 3 ---\n")
            f.write(" ".join(lcvr_cands[2] if len(lcvr_cands) > 2 else lcvr_cands[0]) + "\n")

    agg = {k: aggregate(per_target_avg[k]) for k in per_target_avg}
    print("\n=== aggregate (mean over 10 targets, each averaged over 50 cand) ===\n")
    keys = ["head_special", "head_eos_misalign", "tail_outsider", "head_uniq"]
    print("config".ljust(20) + "".join(k.ljust(22) for k in keys))
    for cfg, m in agg.items():
        cells = [f"{m[k]*100:6.2f}%".ljust(22) for k in keys]
        print(cfg.ljust(20) + "".join(cells))

    # markdown
    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Sonnet baseline vs LCVR\n"]
    lines.append("| config | head_special↓ | head_eos_misalign↓ | tail_outsider↓ | head_uniq↑ |")
    lines.append("|---|---|---|---|---|")
    for cfg, m in agg.items():
        lines.append(
            f"| {cfg} | {m['head_special']*100:.2f}% | {m['head_eos_misalign']*100:.2f}% "
            f"| {m['tail_outsider']*100:.2f}% | {m['head_uniq']*100:.2f}% |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[ok] markdown -> {out_md}")
    print(f"[ok] readable -> {out_dir}/")


if __name__ == "__main__":
    main()
