"""Compare scratch vs sonnet→ci backbone-transfer convergence.

Inputs:
  --scratch_curve <label1>:<ckpt_step>:<decoded_json_path> ...
  --transfer_curve <label1>:<ckpt_step>:<decoded_json_path> ...
  --target_json   target_ci_subset10.json
  --vocab         diffusion_models/.../vocab.json
  --corpus        datasets/ci8w/ci_train.txt
  --out_md        path to write markdown report
  --out_csv       (optional) flat CSV row-per-step

Outputs:
  - markdown: side-by-side per-step structural metrics for scratch vs
    transfer, plus "first step that hits target" rows for each metric
    (steps-to-zero-structural-error).
  - CSV: same data as long-format rows.

Reuses the metric definitions from `compare_lcvr.py` (head_special,
head_punct_misalign, head_zero_freq, head_low_freq, head_common,
head_uniq_ratio, tail_outsider).

Use case for §6.5 of the EMNLP paper:
  python scripts/compare_finetune_vs_scratch.py \
    --scratch_curve \
        scratch_10k:10000:out_gen/.../infill_lcvr_10k.json \
        scratch_25k:25000:out_gen/.../infill_lcvr_25k.json \
        scratch_50k:50000:out_gen/.../infill_lcvr_50k.json \
        scratch_100k:100000:out_gen/.../infill_lcvr_100k.json \
        scratch_200k:200000:out_gen/.../infill_lcvr_200k.json \
        scratch_400k:400000:out_gen/.../infill_lcvr_400k.json \
    --transfer_curve \
        transfer_5k:5000:out_gen/.../infill_lcvr_ft5k.json \
        transfer_10k:10000:out_gen/.../infill_lcvr_ft10k.json \
        transfer_25k:25000:out_gen/.../infill_lcvr_ft25k.json \
        transfer_50k:50000:out_gen/.../infill_lcvr_ft50k.json \
    --target_json control_gen/target_ci_subset10.json \
    --vocab diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/vocab.json \
    --corpus ../datasets/ci8w/ci_train.txt \
    --out_md doc/transfer_curve.md \
    --out_csv doc/transfer_curve.csv
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_lcvr import (  # noqa: E402
    load_lines_as_dicts,
    load_targets,
    load_corpus_freq,
    per_target_metrics,
    aggregate,
)

METRIC_KEYS = [
    "head_special",
    "head_punct_misalign",
    "head_zero_freq",
    "head_low_freq",
    "head_common",
    "head_uniq_ratio",
    "tail_outsider",
]
LOWER_BETTER = {
    "head_special",
    "head_punct_misalign",
    "head_zero_freq",
    "head_low_freq",
    "tail_outsider",
}


def parse_curve_arg(triplets):
    out = []
    for tr in triplets:
        parts = tr.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"bad --*_curve entry (need label:step:path): {tr}")
        label, step_s, path = parts
        out.append((label, int(step_s), Path(path)))
    return out


def metrics_for_ckpt(json_path: Path, targets, char_freq, f_low):
    """Return aggregated metrics for one decoded ckpt."""
    items = load_lines_as_dicts(json_path)
    per = []
    for i, d in enumerate(items):
        if i >= len(targets):
            break
        cand = list(d.values())[0]
        tw = targets[i]["words_"]
        per.append(per_target_metrics(cand, tw, char_freq, f_low))
    return aggregate(per) if per else {k: float("nan") for k in METRIC_KEYS}


def first_step_below(curve, key, threshold):
    """Earliest step at which `key` is <= threshold (or >= for higher-better)."""
    higher_better = key not in LOWER_BETTER
    for label, step, _, met in curve:
        v = met.get(key, float("nan"))
        if np.isnan(v):
            continue
        ok = (v >= threshold) if higher_better else (v <= threshold)
        if ok:
            return step
    return None


def render_md_table(title, curve):
    lines = [f"### {title}", ""]
    head = ["step", "label"] + METRIC_KEYS
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(["---"] * len(head)) + "|")
    for label, step, _, met in curve:
        row = [f"{step:,}", label] + [
            f"{met.get(k, float('nan')):.4f}" for k in METRIC_KEYS
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def render_steps_table(scratch, transfer):
    """Compute steps-to-first-cross for several thresholds."""
    cases = [
        ("head_special", 0.05, "≤ 5%"),
        ("head_special", 0.01, "≤ 1%"),
        ("head_punct_misalign", 0.05, "≤ 5%"),
        ("tail_outsider", 0.10, "≤ 10%"),
        ("tail_outsider", 0.01, "≤ 1%"),
        ("head_uniq_ratio", 0.80, "≥ 80%"),
    ]
    lines = ["### Steps-to-target convergence", ""]
    lines.append("| metric / target | scratch | transfer | speedup |")
    lines.append("|---|---|---|---|")
    for key, thr, human in cases:
        s = first_step_below(scratch, key, thr)
        t = first_step_below(transfer, key, thr)
        if s and t:
            speed = f"{s / t:.1f}×"
        elif s and not t:
            speed = "transfer NA"
        elif t and not s:
            speed = "scratch NA"
        else:
            speed = "—"
        lines.append(
            f"| {key} {human} | "
            f"{f'{s:,}' if s else '—'} | "
            f"{f'{t:,}' if t else '—'} | "
            f"{speed} |"
        )
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch_curve", nargs="+", required=True,
                    help="label:step:decoded_json triplets")
    ap.add_argument("--transfer_curve", nargs="+", required=True)
    ap.add_argument("--target_json", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    targets = load_targets(args.target_json)
    print(f"[info] {len(targets)} targets")

    vocab = json.load(open(args.vocab))
    char_freq, f_low = load_corpus_freq(args.corpus, vocab)
    print(f"[info] f_low (20%-quantile) = {f_low}")

    def build_curve(triplets):
        rows = []
        for label, step, path in triplets:
            if not path.exists():
                print(f"[warn] missing: {path}")
                continue
            met = metrics_for_ckpt(path, targets, char_freq, f_low)
            rows.append((label, step, path, met))
            print(f"  {label:>16}  step={step:>7}  "
                  f"head_sp={met['head_special']:.4f}  "
                  f"tail_out={met['tail_outsider']:.4f}  "
                  f"uniq={met['head_uniq_ratio']:.4f}")
        rows.sort(key=lambda r: r[1])
        return rows

    print("[scratch curve]")
    scratch = build_curve(parse_curve_arg(args.scratch_curve))
    print("[transfer curve]")
    transfer = build_curve(parse_curve_arg(args.transfer_curve))

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Transfer (sonnet → ci) vs scratch — §6.5 evidence\n\n")
        f.write("Decoded with LCVR; same target subset (subset10) for "
                "both groups; metrics defined in `compare_lcvr.py`.\n\n")
        f.write(render_md_table("Scratch ci", scratch))
        f.write("\n")
        f.write(render_md_table("Transfer (sonnet → ci finetune)", transfer))
        f.write("\n")
        f.write(render_steps_table(scratch, transfer))
    print(f"[ok] wrote {out_md}")

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["group", "step", "label"] + METRIC_KEYS)
            for grp, curve in [("scratch", scratch), ("transfer", transfer)]:
                for label, step, _, met in curve:
                    w.writerow(
                        [grp, step, label]
                        + [met.get(k, float("nan")) for k in METRIC_KEYS]
                    )
        print(f"[ok] wrote {out_csv}")


if __name__ == "__main__":
    main()
