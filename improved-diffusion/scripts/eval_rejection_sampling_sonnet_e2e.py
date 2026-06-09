#!/usr/bin/env python
"""
Rejection-Sampling baseline (RS-N) for sonnet/E2E (cross-domain
companion to eval_rejection_sampling_baseline.py for ci).

Re-uses metric functions from sonnet_eval.py / e2e_eval.py to count
how many of the N=50 baseline candidates per target satisfy the
structural constraints
  head_special == 0 AND tail_outsider == 0 AND head_eos_misalign == 0
(or boundary-align == 1 for E2E, where E2E uses a slightly different
schema with a single END at position L).

Compare to LCVR's 100% one-shot acceptance.

Usage:
  python scripts/eval_rejection_sampling_sonnet_e2e.py \
      --domain sonnet \
      --baseline_json out_gen/sonnet_son10_100000.json \
      --target_json control_gen/target_son10.json

  python scripts/eval_rejection_sampling_sonnet_e2e.py \
      --domain e2e \
      --baseline_json out_gen/e2e_e2e30_25000.json \
      --target_json control_gen/target_e2e30.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sonnet_eval import metrics_for_candidate as sonnet_metrics, \
    load_targets as sonnet_load_targets


def is_sonnet_clean(metrics):
    return (metrics["head_special"] == 0.0
            and metrics["tail_outsider"] == 0.0
            and metrics["head_eos_misalign"] == 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["sonnet", "e2e"])
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--target_json", required=True)
    args = ap.parse_args()

    with open(args.baseline_json) as f:
        data = json.load(f)
    if "baseline" not in data:
        print(f"[err] no 'baseline' key in {args.baseline_json}")
        sys.exit(1)
    baseline_cands = data["baseline"]   # list of candidates (token lists)

    targets = sonnet_load_targets(Path(args.target_json))

    # baseline format: list of single token-lists (1 cand per target?)
    # Or nested: list of N targets × M candidates?
    # Sonnet baseline: "baseline": [[tok,...], ...] — each row 1 candidate.
    # Check if grouping is per-target or flat.
    print(f"[info] baseline count = {len(baseline_cands)}, "
          f"targets = {len(targets)}")
    n_per_target = len(baseline_cands) // len(targets) if len(targets) else 0
    if n_per_target == 0:
        print("[err] no baselines / target mismatch")
        sys.exit(1)
    print(f"[info] per-target candidates = {n_per_target}")

    n_targets = len(targets)
    n_total_cands = 0
    n_clean_cands = 0
    n_head_special_cands = 0
    n_tail_outsider_cands = 0
    n_misalign_cands = 0
    accept_per_target = []

    for ti, target in enumerate(targets):
        idx0 = ti * n_per_target
        cleans = 0
        for ci in range(n_per_target):
            cand = baseline_cands[idx0 + ci]
            if not isinstance(cand, list):
                continue
            if args.domain == "sonnet":
                m = sonnet_metrics(cand, target)
            else:  # e2e — define a simple proxy here
                # e2e schema: target has 'L' position; head_special on
                # positions [0, L); tail_outsider on (L, T); boundary
                # at L should be END.
                L = target.get("L") or len(target.get("tokens", []))
                T = len(cand)
                head_specials = sum(1 for w in cand[:L]
                                    if w in {"START", "END", "UNK", "PAD",
                                             "STOP"})
                tail_outsiders = sum(1 for w in cand[L+1:T]
                                     if w not in {"END", "PAD"})
                boundary_ok = (T > L and cand[L] == "END")
                m = dict(head_special=head_specials/max(1, L),
                         head_eos_misalign=0.0 if boundary_ok else 1.0,
                         tail_outsider=tail_outsiders/max(1, T-L-1))
            n_total_cands += 1
            if m["head_special"] > 0:
                n_head_special_cands += 1
            if m["tail_outsider"] > 0:
                n_tail_outsider_cands += 1
            if m["head_eos_misalign"] > 0:
                n_misalign_cands += 1
            if is_sonnet_clean(m) if args.domain == "sonnet" else \
               (m["head_special"] == 0 and m["tail_outsider"] == 0
                and m["head_eos_misalign"] == 0):
                cleans += 1
                n_clean_cands += 1
        accept_per_target.append(cleans / max(1, n_per_target))

    n_targets_with_any = sum(1 for r in accept_per_target if r > 0)

    print(f"\n=== {args.domain} RS-N baseline (rejection sampling) ===")
    print(f"file = {Path(args.baseline_json).name}")
    print(f"per-candidate clean   : {n_clean_cands}/{n_total_cands} "
          f"= {100.0*n_clean_cands/max(n_total_cands,1):.2f}%")
    print(f"per-candidate failures:")
    print(f"  any head_special      : {n_head_special_cands}/{n_total_cands} "
          f"= {100.0*n_head_special_cands/max(n_total_cands,1):.2f}%")
    print(f"  any tail_outsider     : {n_tail_outsider_cands}/{n_total_cands} "
          f"= {100.0*n_tail_outsider_cands/max(n_total_cands,1):.2f}%")
    print(f"  template misalign     : {n_misalign_cands}/{n_total_cands} "
          f"= {100.0*n_misalign_cands/max(n_total_cands,1):.2f}%")
    print(f"per-target accept (any clean): "
          f"{n_targets_with_any}/{n_targets} "
          f"= {100.0*n_targets_with_any/max(n_targets,1):.2f}%")
    print(f"mean per-target accept rate : "
          f"{100.0*sum(accept_per_target)/max(len(accept_per_target),1):.2f}%")


if __name__ == "__main__":
    main()
