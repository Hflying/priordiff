#!/usr/bin/env python
"""
Batch eval driver for CI transfer-curve baseline (§6.5).

For each EMA step in --steps and each variant in {baseline, lcvr_no_freq,
lcvr_drop_low_zero}, compute the {head_special, head_uniq, head_common,
tail_outsider} metrics by reusing scripts/compare_lcvr.per_target_metrics
on the corresponding *.json output of infill.py / decode_pkl_lcvr.py.

Outputs both a console table and a markdown table appended to
  doc/crossdomain_baseline_table.md
under a freshly added section.

Usage:
  python scripts/eval_ci_transfer_curve.py \
    --steps 25 50 75 100 \
    --run_name diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu \
    --tag from_scratch
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the metric computation from the existing comparator.
from compare_lcvr import (   # noqa: E402
    load_targets,
    load_corpus_freq,
    load_lines_as_dicts,
    per_target_metrics,
    aggregate,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", nargs="+", type=int, default=[25, 50, 75, 100],
                   help="EMA step counts in thousands (e.g. 25 50 75 100)")
    p.add_argument("--run_name", required=True,
                   help="diffusion_models subdir, e.g. "
                        "diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu")
    p.add_argument("--tag", default="from_scratch",
                   help="row tag in the markdown table")
    p.add_argument("--target_json",
                   default=str(ROOT / "control_gen/target_ci_subset10.json"))
    p.add_argument("--vocab_json", default=None,
                   help="vocab.json path; default = inside the run dir")
    p.add_argument("--corpus", default=str(ROOT.parent /
                                            "datasets/ci8w/ci_train.txt"))
    p.add_argument("--out_md", default=str(ROOT.parent /
                                           "doc/crossdomain_baseline_table.md"))
    p.add_argument("--variants", nargs="+",
                   default=["baseline", "lcvr_no_freq", "lcvr_drop_low_zero"],
                   help="which variants to score per step")
    return p.parse_args()


METRIC_NAMES = ["head_special", "head_uniq_ratio", "head_common",
                "tail_outsider"]


def file_for(run_name, step_k, variant):
    out_dir = ROOT / "out_gen/control_tone_vowel_length"
    if variant == "baseline":
        # naming used by ci_control_tone_vowel_sz12.sh
        # INFILL_NOTES=tone_vowel_length_sz12_subset10_step{N}k_bfreeze
        return (out_dir /
                f"infill_tone_vowel_length_sz12_subset10_step{step_k}k_"
                f"bfreeze_subset10.json")
    if variant == "lcvr_no_freq":
        return out_dir / f"infill_lcvr_{step_k}k_bfreeze_no_freq.json"
    if variant == "lcvr_drop_low_zero":
        return out_dir / f"infill_lcvr_{step_k}k_bfreeze_drop_low_zero.json"
    raise ValueError(variant)


def main():
    args = parse_args()
    run_dir = ROOT / "diffusion_models" / args.run_name
    vocab_path = (Path(args.vocab_json) if args.vocab_json
                  else run_dir / "vocab.json")
    if not vocab_path.exists():
        print(f"[err] vocab not found: {vocab_path}", file=sys.stderr)
        sys.exit(1)
    vocab = json.load(open(vocab_path))
    targets = load_targets(args.target_json)
    char_freq, f_low = load_corpus_freq(args.corpus, vocab)
    print(f"[info] {len(targets)} targets, f_low={f_low}")

    rows = {}   # (step_k, variant) -> agg
    for step_k in args.steps:
        for variant in args.variants:
            path = file_for(args.run_name, step_k, variant)
            if not path.exists():
                print(f"[warn] missing {path.name}")
                rows[(step_k, variant)] = None
                continue
            items = load_lines_as_dicts(path)
            per_target = []
            for i, d in enumerate(items):
                cand = list(d.values())[0]
                tw = targets[i]["words_"]
                m = per_target_metrics(cand, tw, char_freq, f_low)
                per_target.append(m)
            agg = aggregate(per_target)
            rows[(step_k, variant)] = agg

    # Console table
    print()
    print(f"{'step':>5} {'variant':<22} "
          + " ".join(f"{m:>20}" for m in METRIC_NAMES))
    for step_k in args.steps:
        for variant in args.variants:
            agg = rows[(step_k, variant)]
            if agg is None:
                line = (f"{step_k:>5}k {variant:<22} "
                        + " ".join(f"{'(missing)':>20}"
                                   for _ in METRIC_NAMES))
            else:
                line = (f"{step_k:>5}k {variant:<22} "
                        + " ".join(f"{agg[m]*100:>19.2f}%"
                                   for m in METRIC_NAMES))
            print(line)

    # Markdown table appended
    md = []
    md.append(f"\n\n## CI transfer-curve baseline ({args.tag})\n")
    md.append("| step | variant | head_special↓ | head_uniq↑ | "
              "head_common↑ | tail_outsider↓ |\n")
    md.append("|---|---|---|---|---|---|\n")
    for step_k in args.steps:
        for variant in args.variants:
            agg = rows[(step_k, variant)]
            if agg is None:
                md.append(f"| {step_k}k | {variant} | (missing) | (missing) | "
                          "(missing) | (missing) |\n")
            else:
                md.append(f"| {step_k}k | {variant} | "
                          f"{agg['head_special']*100:.2f}% | "
                          f"{agg['head_uniq_ratio']*100:.2f}% | "
                          f"{agg['head_common']*100:.2f}% | "
                          f"{agg['tail_outsider']*100:.2f}% |\n")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "a") as f:
        f.write("".join(md))
    print(f"[info] appended markdown to {args.out_md}")


if __name__ == "__main__":
    main()
