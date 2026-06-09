#!/usr/bin/env python
"""
Post-hoc token-level constrained-decoding baseline for §7 ablation.

This is a *direct* adaptation of the standard GBS / LM Format Enforcer
spirit to a continuous Diffusion-LM: we take the **unconstrained
baseline argmax output** (the 50-candidate JSON we already have from
the saved `.pkl` latents), and for each candidate, perform a **token-
level substitution pass** that enforces a subset of the LCVR template:

  level 1  (no_pos): replace any special / punct token in head_slots
                     with the corpus-modal char token; replace any
                     non-{END,PAD} token in the tail with PAD; the
                     character vocabulary is shared across positions,
                     so this is the *token-level* analogue of LCVR-
                     no_pos.

  level 2  (tpl_nopunct): same as level 1, but allow punctuations only
                     at punct positions in the template.

  level 3  (full):   full LCVR template (punct slot ⇔ punct only; char
                     slot ⇔ char only; boundary = END; tail = {END,PAD}).

Contrast with LCVR:
  - LCVR operates on the *logits* of the diffusion head, before argmax.
  - This baseline operates on the *token sequence* produced by
    unconstrained argmax.
  - Both reduce to the same V_i conceptually, but the token-level pass
    has no access to the logit scores and therefore cannot pick a
    "best-within-V_i" token — it picks the "corpus-modal char" as a
    1-size approximation.

This shows the value of the latent-level operation: if LCVR
beats this baseline on head_uniq / head_common, the gain is
attributable to using the latent scores to pick *informed*
replacements, not just to enforcing V_i.

Usage:
  python scripts/eval_token_level_posthoc_baseline.py \\
    --baseline_json out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_step400k_bfreeze_subset10.json \\
    --target_json control_gen/target_ci_subset10.json \\
    --corpus ../datasets/ci8w/ci_train.txt \\
    --vocab diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/vocab.json \\
    --level full \\
    --out out_gen/control_tone_vowel_length/baseline_tokenlevel_400k_full.json
"""
import argparse
import ast
import json
from collections import Counter
from pathlib import Path
import sys


SPECIAL = {"START", "END", "PAD", "UNK", "STOP"}
PUNCT_CI = {"，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）",
            ",", ".", "?", "!", ";", ":"}


def load_lines_python_repr(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(ast.literal_eval(line))
    return out


def load_targets(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def corpus_modal_char(corpus_path, vocab):
    """Return the most-frequent non-special, non-punct token in the corpus
    that is also present in vocab.  Used as the fallback char for token-
    level substitution."""
    cnt = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|", 1)
            text = parts[1] if len(parts) >= 2 else parts[0]
            for ch in text:
                if ch in SPECIAL or ch in PUNCT_CI or ch.isspace():
                    continue
                if ch in vocab:
                    cnt[ch] += 1
    if not cnt:
        return "的"
    return cnt.most_common(1)[0][0]


def corpus_modal_punct(corpus_path, vocab):
    """Most-frequent punct in corpus (for punct slots when baseline
    produced specials instead)."""
    cnt = Counter()
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|", 1)
            text = parts[1] if len(parts) >= 2 else parts[0]
            for ch in text:
                if ch in PUNCT_CI and ch in vocab:
                    cnt[ch] += 1
    if not cnt:
        return "，"
    return cnt.most_common(1)[0][0]


def token_level_substitute(cand_str, target_words, modal_char, modal_punct,
                           level):
    """Apply the token-level substitution pass to a single candidate."""
    L = len(target_words)
    is_punct = [w in PUNCT_CI for w in target_words]
    toks = cand_str.split()
    T = len(toks)
    out_toks = list(toks)

    for i in range(T):
        if i < L:
            t = out_toks[i]
            if level == "no_pos":
                if t in SPECIAL or t in PUNCT_CI:
                    out_toks[i] = modal_char
            elif level == "tpl_nopunct":
                if t in SPECIAL:
                    out_toks[i] = modal_char if not is_punct[i] else modal_punct
                elif is_punct[i] and t not in PUNCT_CI:
                    # punct slot but baseline picked non-punct — keep it
                    # (tpl_nopunct does not enforce punct-positions)
                    pass
            else:  # full
                if is_punct[i]:
                    # punct slot: force to a punct token
                    if t in SPECIAL or t not in PUNCT_CI:
                        out_toks[i] = modal_punct
                else:
                    # char slot: force to a non-special, non-punct token
                    if t in SPECIAL or t in PUNCT_CI:
                        out_toks[i] = modal_char
        elif i == L:
            out_toks[i] = "END"
        else:
            if out_toks[i] not in ("END", "PAD"):
                out_toks[i] = "PAD"
    return " ".join(out_toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_json", required=True)
    ap.add_argument("--target_json", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--level", default="full",
                    choices=["no_pos", "tpl_nopunct", "full"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    vocab = json.load(open(args.vocab))
    print(f"[info] vocab size = {len(vocab)}")

    modal_char = corpus_modal_char(args.corpus, vocab)
    modal_punct = corpus_modal_punct(args.corpus, vocab)
    print(f"[info] modal_char = '{modal_char}', modal_punct = '{modal_punct}'")
    print(f"[info] level = {args.level}")

    targets = load_targets(args.target_json)
    items = load_lines_python_repr(args.baseline_json)
    assert len(items) == len(targets), f"{len(items)} items vs {len(targets)} targets"

    out_items = []
    for i, d in enumerate(items):
        k = list(d.keys())[0]
        cand_list = list(d.values())[0]
        new_cand = [token_level_substitute(c, targets[i]["words_"],
                                            modal_char, modal_punct,
                                            args.level)
                    for c in cand_list]
        out_items.append({k: new_cand})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for d in out_items:
            # emit as one python-repr-dict per line for compatibility with
            # compare_lcvr.py / eval_rejection_sampling_baseline.py
            f.write(repr(d) + "\n")
    print(f"[done] wrote {len(out_items)} entries to {args.out}")


if __name__ == "__main__":
    main()
