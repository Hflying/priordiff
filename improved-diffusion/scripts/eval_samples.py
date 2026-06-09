"""
Quick evaluation of generated samples against the ci8w dataset.

Reports:
  - format    : avg lines per poem, avg chars per line, length stats
  - structure : START/END separator compliance, OOV / UNK ratio
  - tonal     : Ping (平) / Ze (仄) ratio at line-end vs reference
  - rhyme     : line-end vowel-group consistency within each poem
  - lexicon   : Distinct-1, Distinct-2, repeat ratio

Usage:
  python scripts/eval_samples.py \
      generation_outputs/diff_e2e-tgt_block_2000steps_16dim_ci_gpu.ema_0.9999_050000.pt.samples_-1.0.json \
      --vocab ../datasets/ci8w/ci_vocab.txt \
      --tone  ../datasets/ci8w/ci_tone.txt \
      --vowel ../datasets/ci8w/ci_vowel.txt \
      [--ref ../datasets/ci8w/ci_valid.txt --ref_n 500]
"""
import argparse
import json
import os
import re
from collections import Counter
from statistics import mean, median, pstdev

LINE_SPLIT_RE = re.compile(r"[，。、；？！]")
PUNCT_SET = set("，。、；？！,.;?!")
SEP_TOKENS = {"START", "END", "<bos>", "<eos>", "PAD", "UNK"}


def load_label_map(vocab_path, label_path):
    """Pair each line of vocab with the corresponding label line."""
    with open(vocab_path, encoding="utf-8") as f_v:
        vocab = [w.strip() for w in f_v]
    with open(label_path, encoding="utf-8") as f_l:
        labels = [l.strip() for l in f_l]
    n = min(len(vocab), len(labels))
    return {vocab[i]: labels[i] for i in range(n)}


def load_samples(path):
    """Returns list[list[str]] — one list of tokens per sample."""
    with open(path, encoding="utf-8") as f:
        first = f.readline().strip()
        f.seek(0)
        if first.startswith("["):
            samples = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, list) and obj and isinstance(obj[0], list):
                    obj = obj[0]
                if len(obj) == 1 and isinstance(obj[0], str) and " " in obj[0]:
                    samples.append(obj[0].split())
                elif isinstance(obj, str):
                    samples.append(obj.split())
                else:
                    samples.append([str(t) for t in obj])
            return samples
        # plain text fallback
        return [line.strip().split() for line in open(path, encoding="utf-8") if line.strip()]


def split_lines(tokens):
    """Split a token sequence into lines (句) using punctuation + START/END markers."""
    lines, cur = [], []
    for t in tokens:
        if t in SEP_TOKENS or t in {"START", "END"}:
            if cur:
                lines.append(cur)
                cur = []
            continue
        if t in PUNCT_SET:
            if cur:
                lines.append(cur)
                cur = []
            continue
        cur.append(t)
    if cur:
        lines.append(cur)
    return [ln for ln in lines if ln]


def evaluate(samples, tone_map, vowel_map, ref_lines=None):
    n = len(samples)

    # ---- format / length ----
    lines_per_poem, chars_per_line, total_chars = [], [], 0
    for toks in samples:
        ls = split_lines(toks)
        lines_per_poem.append(len(ls))
        chars_per_line.extend(len(ln) for ln in ls)
        total_chars += sum(len(ln) for ln in ls)

    # ---- separator compliance ----
    have_sep = sum(1 for toks in samples if "START" in toks or "END" in toks)

    # ---- vocabulary coverage ----
    char_counter = Counter()
    oov = 0
    unk = 0
    for toks in samples:
        for t in toks:
            if t in PUNCT_SET or t in SEP_TOKENS:
                continue
            char_counter[t] += 1
            if t == "UNK":
                unk += 1
            elif t not in tone_map:
                oov += 1

    # ---- tone (line-end) ----
    end_tone = Counter()
    for toks in samples:
        for ln in split_lines(toks):
            last = ln[-1]
            end_tone[tone_map.get(last, "UNK")] += 1

    # ---- rhyme (line-end vowel agreement within a poem) ----
    rhyme_consistency = []
    for toks in samples:
        ls = split_lines(toks)
        end_vowels = []
        for ln in ls:
            v = vowel_map.get(ln[-1], "")
            primary = v.split()[0] if v else ""
            if primary:
                end_vowels.append(primary)
        if len(end_vowels) >= 2:
            most = Counter(end_vowels).most_common(1)[0][1]
            rhyme_consistency.append(most / len(end_vowels))

    # ---- diversity ----
    unigrams = []
    bigrams = []
    for toks in samples:
        clean = [t for t in toks if t not in SEP_TOKENS]
        unigrams.extend(clean)
        bigrams.extend(zip(clean, clean[1:]))
    distinct1 = len(set(unigrams)) / max(1, len(unigrams))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))

    # ---- reference comparison (optional) ----
    ref_summary = None
    if ref_lines is not None:
        ref_chars_per_line = []
        ref_end_tone = Counter()
        ref_rhyme_consistency = []
        for line in ref_lines:
            # ci_train.txt format: "TUNE|content"
            content = line.split("|", 1)[-1]
            toks = [c for c in content.strip()]
            ls = split_lines(toks)
            ref_chars_per_line.extend(len(ln) for ln in ls)
            ev = []
            for ln in ls:
                last = ln[-1]
                ref_end_tone[tone_map.get(last, "UNK")] += 1
                v = vowel_map.get(last, "")
                primary = v.split()[0] if v else ""
                if primary:
                    ev.append(primary)
            if len(ev) >= 2:
                most = Counter(ev).most_common(1)[0][1]
                ref_rhyme_consistency.append(most / len(ev))
        ref_summary = dict(
            chars_per_line_avg=mean(ref_chars_per_line) if ref_chars_per_line else 0,
            end_tone_PING=ref_end_tone.get("PING", 0)
            / max(1, sum(ref_end_tone.values())),
            end_tone_ZE=ref_end_tone.get("ZE", 0) / max(1, sum(ref_end_tone.values())),
            rhyme_consistency=mean(ref_rhyme_consistency)
            if ref_rhyme_consistency
            else 0,
        )

    # ---- print ----
    print("=" * 60)
    print(f"samples evaluated : {n}")
    print(f"avg lines / poem  : {mean(lines_per_poem):.2f}  (median {median(lines_per_poem)})")
    print(f"avg chars / line  : {mean(chars_per_line):.2f}  (std {pstdev(chars_per_line):.2f})")
    print(f"START/END coverage: {have_sep / n:.1%}")
    print(f"UNK / OOV ratio   : UNK={unk / max(1, total_chars):.2%}, OOV={oov / max(1, total_chars):.2%}")
    print()

    tone_total = max(1, sum(end_tone.values()))
    print("--- 平仄 (line-end tone) ---")
    print(f"  PING  (平): {end_tone['PING'] / tone_total:.1%}")
    print(f"  ZE    (仄): {end_tone['ZE'] / tone_total:.1%}")
    print(f"  NONE      : {end_tone.get('NONE', 0) / tone_total:.1%}")
    print(f"  UNK       : {end_tone.get('UNK', 0) / tone_total:.1%}")
    print()

    if rhyme_consistency:
        print(f"--- 韵脚一致率 (within-poem) ---")
        print(f"  mean : {mean(rhyme_consistency):.3f}   (1.0 = perfect rhyme)")
        print(f"  perfect-rhyme poems: {sum(1 for r in rhyme_consistency if r == 1.0) / len(rhyme_consistency):.1%}")
        print()

    print("--- 多样性 (lexical diversity) ---")
    print(f"  Distinct-1: {distinct1:.3f}")
    print(f"  Distinct-2: {distinct2:.3f}")
    print(f"  unique chars used: {len(char_counter)}")
    print()

    if ref_summary:
        print("--- 参考集（同样指标，用真实 ci8w valid 集对比）---")
        for k, v in ref_summary.items():
            print(f"  {k:25s}: {v:.3f}" if isinstance(v, float) else f"  {k:25s}: {v}")
        print()

    print("--- 高频字 top 20 ---")
    for c, k in char_counter.most_common(20):
        print(f"  {c}  {k}")
    print("=" * 60)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("samples", help="path to *.json or *.txt sample file")
    p.add_argument("--vocab", required=True)
    p.add_argument("--tone", required=True)
    p.add_argument("--vowel", required=True)
    p.add_argument("--ref", default=None, help="optional reference .txt to compare")
    p.add_argument("--ref_n", type=int, default=500, help="how many ref lines to use")
    args = p.parse_args()

    tone_map = load_label_map(args.vocab, args.tone)
    vowel_map = load_label_map(args.vocab, args.vowel)
    samples = load_samples(args.samples)

    ref_lines = None
    if args.ref and os.path.isfile(args.ref):
        with open(args.ref, encoding="utf-8") as f:
            ref_lines = [next(f) for _ in range(args.ref_n)]

    evaluate(samples, tone_map, vowel_map, ref_lines=ref_lines)


if __name__ == "__main__":
    main()
