#!/usr/bin/env python
"""
Rejection-Sampling baseline (RS-N) for §6.1 / §6.2 / §6.3 / §6.4.

For a given baseline (unconstrained-argmax) infill_*.json file with
N=50 candidates per target, count how many of the 50 candidates pass
the structural constraints

  head_special == 0  AND
  tail_outsider == 0 AND
  boundary_align == 1   (token at position L equals END)

This is the natural autoregressive baseline ported to continuous
diffusion: re-sample until success.  If our LCVR claim is right, we
expect the per-candidate acceptance rate to be ~5-30% on the
worst-affected domain (ci-block, BAD-heavy late checkpoints) and
~50-95% on cleaner pad-mode domains, vs LCVR's 100% one-shot
acceptance on every domain.

Usage:
  # ci-400k baseline (original Diffusion-LM checkpoint)
  python scripts/eval_rejection_sampling_baseline.py \
    --baseline_json out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_step400k_bfreeze_subset10.json \
    --target_json control_gen/target_ci_subset10.json \
    --domain ci

  # ci-25k (early-training, pre-BAD)
  python scripts/eval_rejection_sampling_baseline.py \
    --baseline_json out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_step25k_bfreeze_subset10.json \
    --target_json control_gen/target_ci_subset10.json \
    --domain ci
"""

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))


SPECIAL = {"START", "END", "PAD", "UNK", "STOP"}
PUNCT = {"，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）",
         ",", ".", "?", "!", ";", ":"}
NEUTRAL_TAIL = {"END", "PAD"}


def load_targets(path):
    targets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    return targets


def load_lines_as_dicts(path):
    """infill 输出是 'one python-repr dict per line' (tuple keys, single quotes)."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(ast.literal_eval(line))
    return items


def candidate_is_clean(cand_str, target_words):
    """Check whether one candidate satisfies all structural constraints.

    Returns:
      (is_clean: bool, head_specials: int, tail_outsiders: int,
       boundary_ok: bool)
    """
    L = len(target_words)
    is_punct_target = [w in PUNCT for w in target_words]
    toks = cand_str.split()

    # head: positions [0, L)
    head_specials = 0
    head_misaligns = 0
    for i in range(min(L, len(toks))):
        t = toks[i]
        if t in SPECIAL:
            head_specials += 1
            continue
        if is_punct_target[i]:
            continue  # punct slot, OK whether or not the model produced punct
        # char slot
        if t in PUNCT:
            head_misaligns += 1

    # boundary: position L should be END
    boundary_ok = (len(toks) > L and toks[L] == "END")

    # tail: positions (L, T)
    tail_outsiders = 0
    for i in range(L + 1, len(toks)):
        t = toks[i]
        if t in SPECIAL and t not in NEUTRAL_TAIL:
            tail_outsiders += 1
        elif t not in SPECIAL and t not in PUNCT:
            tail_outsiders += 1

    is_clean = (head_specials == 0
                and head_misaligns == 0
                and tail_outsiders == 0
                and boundary_ok)
    return is_clean, head_specials, tail_outsiders, boundary_ok


def head_uniq_for_clean(cand_str, target_words):
    """Compute head_uniq_ratio over the (clean) candidate's char slots."""
    L = len(target_words)
    is_punct_target = [w in PUNCT for w in target_words]
    toks = cand_str.split()
    chars = []
    for i in range(min(L, len(toks))):
        t = toks[i]
        if t in SPECIAL or is_punct_target[i] or t in PUNCT:
            continue
        chars.append(t)
    if not chars:
        return None
    return len(set(chars)) / len(chars)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--target_json", required=True)
    ap.add_argument("--domain", default="ci",
                    choices=["ci", "sonnet", "e2e"],
                    help="affects target_json schema")
    ap.add_argument("--out_md", default=None,
                    help="if set, append a markdown row")
    args = ap.parse_args()

    targets = load_targets(args.target_json)
    items = load_lines_as_dicts(args.baseline_json)
    if len(items) != len(targets):
        print(f"[warn] {len(items)} items vs {len(targets)} targets")

    n_targets = len(targets)
    accept_per_target = []
    head_uniq_clean_per_target = []
    n_total_cands = 0
    n_clean_cands = 0
    n_head_special_cands = 0
    n_tail_outsider_cands = 0
    n_boundary_fail_cands = 0

    for i, d in enumerate(items):
        cand = list(d.values())[0]
        if args.domain == "ci":
            tw = targets[i]["words_"]
        elif args.domain == "sonnet":
            # sonnet jsonl: keys 'tokens' or 'words'
            tw = targets[i].get("tokens") or targets[i].get("words")
        else:
            tw = targets[i].get("words_") or targets[i].get("tokens")

        cleans = []
        huqs = []
        for s in cand:
            ok, hs, to, bo = candidate_is_clean(s, tw)
            n_total_cands += 1
            if hs > 0:
                n_head_special_cands += 1
            if to > 0:
                n_tail_outsider_cands += 1
            if not bo:
                n_boundary_fail_cands += 1
            if ok:
                n_clean_cands += 1
                cleans.append(s)
                hu = head_uniq_for_clean(s, tw)
                if hu is not None:
                    huqs.append(hu)

        accept_rate = len(cleans) / len(cand) if cand else 0.0
        accept_per_target.append(accept_rate)
        if huqs:
            head_uniq_clean_per_target.append(sum(huqs) / len(huqs))

    n_targets_with_any_accept = sum(1 for r in accept_per_target if r > 0)

    print(f"\n=== Rejection-Sampling baseline ===")
    print(f"file = {Path(args.baseline_json).name}")
    print(f"targets = {n_targets}, candidates = {n_total_cands}")
    print(f"per-candidate clean   : {n_clean_cands}/{n_total_cands} "
          f"= {100.0*n_clean_cands/max(n_total_cands,1):.2f}%")
    print(f"per-candidate failures:")
    print(f"  any head_special      : {n_head_special_cands}/{n_total_cands} "
          f"= {100.0*n_head_special_cands/max(n_total_cands,1):.2f}%")
    print(f"  any tail_outsider     : {n_tail_outsider_cands}/{n_total_cands} "
          f"= {100.0*n_tail_outsider_cands/max(n_total_cands,1):.2f}%")
    print(f"  boundary not END at L : {n_boundary_fail_cands}/{n_total_cands} "
          f"= {100.0*n_boundary_fail_cands/max(n_total_cands,1):.2f}%")
    print(f"per-target acceptance (any clean cand): "
          f"{n_targets_with_any_accept}/{n_targets} "
          f"= {100.0*n_targets_with_any_accept/max(n_targets,1):.2f}%")
    print(f"mean per-target accept rate: "
          f"{100.0*sum(accept_per_target)/max(len(accept_per_target),1):.2f}%")
    if head_uniq_clean_per_target:
        mean_huq = sum(head_uniq_clean_per_target) / len(head_uniq_clean_per_target)
        print(f"head_uniq on clean cands : {100.0*mean_huq:.2f}% "
              f"(over {len(head_uniq_clean_per_target)} targets that had ≥1 clean)")
    else:
        mean_huq = None
        print(f"head_uniq on clean cands : N/A (no clean candidates anywhere)")

    if args.out_md is not None:
        line = (f"| {Path(args.baseline_json).name} | "
                f"{100.0*n_clean_cands/max(n_total_cands,1):.2f}% | "
                f"{100.0*sum(accept_per_target)/max(len(accept_per_target),1):.2f}% | "
                f"{100.0*n_targets_with_any_accept/max(n_targets,1):.2f}% | "
                f"{100.0*mean_huq:.2f}% |\n"
                if mean_huq is not None else
                f"| {Path(args.baseline_json).name} | "
                f"{100.0*n_clean_cands/max(n_total_cands,1):.2f}% | "
                f"{100.0*sum(accept_per_target)/max(len(accept_per_target),1):.2f}% | "
                f"{100.0*n_targets_with_any_accept/max(n_targets,1):.2f}% | N/A |\n")
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, "a") as f:
            f.write(line)
        print(f"[info] appended to {args.out_md}")


if __name__ == "__main__":
    main()
