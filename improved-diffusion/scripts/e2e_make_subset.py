"""从 e2e_data/src1_valid.txt 抽 N 条作为评测 subset 并提取 slot 信息。

每条记录：
  {
    "subset_idx", "src_idx",
    "mr_str"      : 原始 MR 字符串
    "slots"       : {key: value, ...}
    "target_str"  : 真值 utterance
    "target_toks" : spaCy 切分后的 token 列表
    "value_tokens": [(slot_key, [value_token list]), ...]   # 每个 slot value 的 token 序列
  }

输出：
  control_gen/target_e2e30.jsonl  (默认 30 条)
  control_gen/target_e2e30_titles.txt
"""

from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_mr(s: str):
    slots = {}
    for kv in s.split("|"):
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        slots[k.strip()] = v.strip()
    return slots


_TOK = None


def tokenize_en(s: str):
    global _TOK
    if _TOK is None:
        from spacy.lang.en import English
        _TOK = English().tokenizer
    return [t.text for t in _TOK(s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--src", default="../datasets/e2e_data/src1_valid.txt")
    ap.add_argument("--out", default="control_gen/target_e2e30.jsonl")
    ap.add_argument("--titles", default="control_gen/target_e2e30_titles.txt")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    src_path = (ROOT / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src)
    out_path = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    titles_path = (ROOT / args.titles).resolve() if not Path(args.titles).is_absolute() else Path(args.titles)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)

    with open(src_path, "r", encoding="utf-8") as f:
        all_lines = [ln.rstrip("\n") for ln in f if ln.strip() and "||" in ln]
    print(f"[info] {len(all_lines)} lines in {src_path}")

    rows = []
    for j, line in enumerate(all_lines):
        mr_str, target = line.split("||", 1)
        slots = parse_mr(mr_str)
        target_toks = tokenize_en(target)
        if len(target_toks) > 64:
            continue
        value_tokens = []
        for k, v in slots.items():
            value_tokens.append((k, tokenize_en(v)))
        rows.append((j, mr_str.strip(), slots, target.strip(), target_toks, value_tokens))

    chosen = random.sample(rows, k=min(args.n, len(rows)))
    chosen.sort(key=lambda x: x[0])

    with open(out_path, "w", encoding="utf-8") as fjson, \
         open(titles_path, "w", encoding="utf-8") as ftit:
        for idx, (src_idx, mr_str, slots, target_str, target_toks, value_tokens) in enumerate(chosen):
            rec = {
                "subset_idx": idx,
                "src_idx": src_idx,
                "mr_str": mr_str,
                "slots": slots,
                "target_str": target_str,
                "target_toks": target_toks,
                "n_tokens": len(target_toks),
                "value_tokens": [list(x) for x in value_tokens],
            }
            fjson.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ftit.write(f"{idx}\t{src_idx}\t{slots.get('name', '?')}\t{slots.get('Type', '?')}\tL={len(target_toks)}\n")
            print(f"  [{idx}] {slots.get('name', '?')} | {slots.get('Type', '?')} | L={len(target_toks)} | slots={len(slots)}")

    print(f"\n[done] {out_path}\n       {titles_path}")


if __name__ == "__main__":
    main()
