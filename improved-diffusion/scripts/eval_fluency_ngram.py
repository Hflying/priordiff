#!/usr/bin/env python
"""
Automated fluency proxy via n-gram corpus overlap.

For each evaluated method on each domain, we compute:
  (a) trigram coverage rate
        = fraction of generated trigrams that occur ≥1 in the
          training corpus
  (b) bigram coverage rate (sanity baseline)
  (c) per-candidate trigram-cross-entropy (Laplace-smoothed)

The unit of analysis is the head-region of each candidate (the
content slot, with specials/punct removed for ci, with the
\\<,eos,\\> triplets removed for sonnet, and with content tokens
only for E2E).  Scores are aggregated per-target (mean over
candidates) then per-method (mean over targets).

Higher coverage = candidate uses more in-domain n-grams = higher
fluency.  This is not a substitute for human evaluation, but
provides a *cheap, automated, reproducible* fluency signal that
distinguishes raw baseline from \\lcvr-projected output.

Usage:
  # ci (5 ckpts × 3 methods)
  python scripts/eval_fluency_ngram.py \\
      --domain ci --corpus ../datasets/ci8w/ci_train.txt \\
      --target_json control_gen/target_ci_subset10.json \\
      --files \\
        baseline_400k=out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_step400k_bfreeze_subset10.json \\
        lcvr_400k=out_gen/control_tone_vowel_length/infill_lcvr_400k_bfreeze_no_freq.json \\
        m3v1_400k=out_gen/control_tone_vowel_length/infill_m3v1_400k.json
"""
from __future__ import annotations
import argparse
import ast
import json
import math
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


SPECIAL = {"START", "END", "PAD", "UNK", "STOP",
           "start", "end", "pad", "unk", "stop"}
PUNCT_CI = {"，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）"}
PUNCT_EN = {",", ".", "?", "!", ";", ":", "'", "\"", "(", ")", "-"}
TRIPLET = {"<", "eos", ">", "<eos>"}  # corpus uses "<eos>" literal


def load_lines_python_repr(path):
    """ci infill output is python-repr-per-line."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ast.literal_eval(line))
    return out


def load_targets_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def head_chars_ci(cand_str, target_words):
    """ci: extract char-level head sequence (drop specials & puncts)."""
    L = len(target_words)
    is_punct = [w in PUNCT_CI for w in target_words]
    toks = cand_str.split()
    chars = []
    for i in range(min(L, len(toks))):
        t = toks[i]
        if t in SPECIAL or t in PUNCT_CI:
            continue
        if is_punct[i]:
            continue
        chars.append(t)
    return chars


def head_words_sonnet(cand_tokens, target_rec):
    """sonnet: word slots = positions [0, last_triplet_end] not in triplet."""
    triplet_pos = (set(target_rec["lt_positions"])
                   | set(target_rec["eos_positions"])
                   | set(target_rec["gt_positions"]))
    last_end = max(target_rec["gt_positions"])
    words = []
    for i in range(min(last_end + 1, len(cand_tokens))):
        if i in triplet_pos:
            continue
        t = cand_tokens[i]
        if t in SPECIAL or t in TRIPLET:
            continue
        words.append(t.lower())
    return words


def head_words_e2e(cand_tokens, target_rec):
    # E2E schema uses "target_toks" (list of words including specials/punct);
    # L = number of target tokens before the END boundary.
    target_list = (target_rec.get("target_toks")
                   or target_rec.get("tokens"))
    L = (target_rec.get("L") or
         (len(target_list) if target_list else 0))
    words = []
    for i in range(min(L, len(cand_tokens))):
        t = cand_tokens[i]
        if t in SPECIAL:
            continue
        if t in PUNCT_EN:
            continue
        words.append(t.lower())
    return words


def build_corpus_ngram_sets(corpus_path, mode, n_max=3):
    """Return ngram sets {n: Counter} from the corpus."""
    grams = {n: Counter() for n in range(1, n_max + 1)}
    total_units = 0
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # ci_train.txt is "<title>|<content>"; split off title.
            parts = line.split("|", 1)
            text = parts[1] if len(parts) >= 2 else parts[0]
            if mode == "ci":
                # char-level (drop punct+spec)
                units = [c for c in text
                         if c not in PUNCT_CI and c not in SPECIAL
                         and not c.isspace()]
            else:
                # word-level lowercase, drop punct
                tokens = text.lower().split()
                units = [t for t in tokens
                         if t not in PUNCT_EN and t not in SPECIAL
                         and t not in TRIPLET]
            total_units += len(units)
            for n in range(1, n_max + 1):
                for i in range(len(units) - n + 1):
                    grams[n][tuple(units[i:i + n])] += 1
    return grams, total_units


def coverage_rate(unit_seq, corpus_ngrams, n):
    """fraction of n-grams in unit_seq that occur in corpus."""
    if len(unit_seq) < n:
        return None
    grams = [tuple(unit_seq[i:i + n])
             for i in range(len(unit_seq) - n + 1)]
    if not grams:
        return None
    hits = sum(1 for g in grams if corpus_ngrams[n].get(g, 0) > 0)
    return hits / len(grams)


def laplace_xent(unit_seq, corpus_ngrams, n, V_alpha):
    """Laplace-smoothed n-gram cross-entropy.

    P(w_i | context) = (count(context, w_i) + 1) / (count(context) + V_alpha)
    """
    if len(unit_seq) < n:
        return None
    log_probs = []
    for i in range(len(unit_seq) - n + 1):
        ctx = tuple(unit_seq[i:i + n - 1])
        wi = unit_seq[i + n - 1]
        full = ctx + (wi,)
        c_full = corpus_ngrams[n].get(full, 0)
        c_ctx = (corpus_ngrams[n - 1].get(ctx, 0)
                 if n - 1 >= 1 else
                 sum(corpus_ngrams[1].values()))
        p = (c_full + 1) / (c_ctx + V_alpha)
        log_probs.append(-math.log2(max(p, 1e-30)))
    if not log_probs:
        return None
    return sum(log_probs) / len(log_probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True, choices=["ci", "sonnet", "e2e"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--target_json", required=True)
    ap.add_argument("--files", nargs="+", required=True,
                    help="space-separated tag=path entries")
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    print(f"[info] building n-gram tables from {args.corpus}")
    grams, total = build_corpus_ngram_sets(args.corpus,
                                            mode=args.domain,
                                            n_max=3)
    V = len(grams[1])
    print(f"[info] vocab |units| = {V}, total units = {total:,}")
    print(f"[info] |bigram types| = {len(grams[2]):,}, "
          f"|trigram types| = {len(grams[3]):,}")

    targets = (load_targets_jsonl(args.target_json)
               if args.target_json.endswith(".jsonl") else
               load_targets_jsonl(args.target_json))

    # parse file list
    parsed = []
    for ent in args.files:
        if "=" not in ent:
            print(f"[err] expected tag=path, got {ent}")
            sys.exit(1)
        tag, path = ent.split("=", 1)
        parsed.append((tag, path))

    print()
    print(f"{'tag':<22} {'bi-cov':>10} {'tri-cov':>10} "
          f"{'tri-xent':>12} {'n_cands':>8}")

    md = []
    md.append(f"\n## Fluency proxy ({args.domain})\n")
    md.append("| method | bigram coverage ↑ | trigram coverage ↑ | "
              "tri-xent ↓ | n cand |\n")
    md.append("|---|---|---|---|---|\n")

    for tag, path in parsed:
        path_full = Path(path) if Path(path).is_absolute() else ROOT / path
        if not path_full.exists():
            print(f"{tag:<22}  MISSING ({path_full.name})")
            md.append(f"| {tag} | (missing) | (missing) | (missing) | 0 |\n")
            continue

        # Two file styles: ci/m3v1 = python-repr-per-line; sonnet/e2e =
        # JSON dict with key 'baseline' or 'lcvr' or 'm3v1'.
        if args.domain == "ci":
            items = load_lines_python_repr(str(path_full))
            cands_per_target = []
            for d in items:
                cands_per_target.append(list(d.values())[0])
        else:
            data = json.load(open(path_full))
            # tag suffix decides which key to read
            if tag.startswith("baseline"):
                key = "baseline"
            elif tag.startswith("lcvr"):
                key = "lcvr"
            elif tag.startswith("m3v1"):
                key = "m3v1"
            else:
                key = list(data.keys())[0]
            if key not in data:
                # try guessing
                key = list(data.keys())[0]
            payload = data[key]
            # baseline is a flat list of token-lists; lcvr/m3v1 maps idx->dict
            if isinstance(payload, list):
                # group by target (assumed equal split)
                n_per = len(payload) // len(targets) if targets else 0
                cands_per_target = [
                    payload[i * n_per:(i + 1) * n_per]
                    for i in range(len(targets))
                ]
            else:
                cands_per_target = []
                for ti in range(len(targets)):
                    tk = str(ti)
                    if tk in payload:
                        cs = payload[tk]
                        toks = (cs.get("tokens") if isinstance(cs, dict)
                                else cs)
                        cands_per_target.append(toks if toks else [])
                    else:
                        cands_per_target.append([])

        bi_covs, tri_covs, tri_xents = [], [], []
        n_total = 0
        for ti, cands in enumerate(cands_per_target):
            if not cands:
                continue
            for c in cands:
                if args.domain == "ci":
                    if isinstance(c, list):
                        c = " ".join(c)
                    units = head_chars_ci(c, targets[ti]["words_"])
                elif args.domain == "sonnet":
                    units = head_words_sonnet(c, targets[ti])
                else:
                    units = head_words_e2e(c, targets[ti])
                n_total += 1
                bc = coverage_rate(units, grams, 2)
                tc = coverage_rate(units, grams, 3)
                xe = laplace_xent(units, grams, 3, V)
                if bc is not None:
                    bi_covs.append(bc)
                if tc is not None:
                    tri_covs.append(tc)
                if xe is not None:
                    tri_xents.append(xe)

        bc_mean = sum(bi_covs) / len(bi_covs) if bi_covs else None
        tc_mean = sum(tri_covs) / len(tri_covs) if tri_covs else None
        xe_mean = sum(tri_xents) / len(tri_xents) if tri_xents else None

        bc_str = f"{bc_mean*100:.2f}%" if bc_mean is not None else "N/A"
        tc_str = f"{tc_mean*100:.2f}%" if tc_mean is not None else "N/A"
        xe_str = f"{xe_mean:.2f}" if xe_mean is not None else "N/A"
        print(f"{tag:<22} {bc_str:>10} {tc_str:>10} "
              f"{xe_str:>12} {n_total:>8}")
        md.append(f"| {tag} | {bc_str} | {tc_str} | {xe_str} | "
                  f"{n_total} |\n")

    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_md, "a") as f:
            f.write("".join(md))
        print(f"[info] appended markdown to {args.out_md}")


if __name__ == "__main__":
    main()
