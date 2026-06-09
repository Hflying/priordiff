"""从 sonnet_valid.txt 抽 10 首作为评测 subset。

注意：训练时 spaCy tokenizer 把 `<eos>` 切成 3 个 token：< / eos / >。
故每首 sonnet 在 token 序列中有 **14 × 3 = 42** 个 line-separator 位置。
LCVR 模板按这 42 个位置硬约束：
  - "lt_positions"  : 14 个，必须是 "<"
  - "eos_positions" : 14 个，必须是 "eos"
  - "gt_positions"  : 14 个，必须是 ">"
  - "word_positions": 其余位置，自由词位
  - "tail_positions": 末尾 padding，必须是 _PAD

每条记录：
  {
    "subset_idx", "src_idx", "tokens", "n_tokens",
    "lt_positions", "eos_positions", "gt_positions",
    "lines"  # 14 行（按 <eos> 三元组切分后的 word lists）
  }
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def tokenize_sonnet(line: str):
    try:
        from spacy.lang.en import English
        nlp = English()
        tokenizer = nlp.tokenizer
        return [x.text for x in tokenizer(line.rstrip("\n"))]
    except Exception:
        return line.rstrip("\n").split()


def parse_sonnet(line: str):
    toks = tokenize_sonnet(line)
    # 找 < eos > 三元组的起始位置
    triplet_starts = [
        i for i in range(len(toks) - 2)
        if toks[i] == "<" and toks[i + 1] == "eos" and toks[i + 2] == ">"
    ]
    lt_pos = triplet_starts
    eos_pos = [i + 1 for i in triplet_starts]
    gt_pos = [i + 2 for i in triplet_starts]
    return toks, lt_pos, eos_pos, gt_pos


def main():
    random.seed(123)
    valid_path = ROOT.parent / "datasets/sonnet3355/sonnet_valid.txt"
    out_dir = ROOT / "control_gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "target_son10.json"
    out_titles = out_dir / "target_son10_titles.txt"

    print(f"[info] reading {valid_path}")
    with open(valid_path, "r", encoding="utf-8") as f:
        all_lines = [ln for ln in f if ln.strip()]
    print(f"[info] {len(all_lines)} valid sonnets")

    samples = []
    for j, line in enumerate(all_lines):
        toks, lt, eos, gt = parse_sonnet(line)
        if len(lt) == 14 and len(toks) <= 196:
            samples.append((j, toks, lt, eos, gt))
    print(f"[info] {len(samples)} sonnets fit (14 line-triplets, ≤196 tok)")

    chosen = random.sample(samples, k=min(10, len(samples)))
    chosen.sort(key=lambda x: x[0])

    with open(out_json, "w", encoding="utf-8") as fjson, \
         open(out_titles, "w", encoding="utf-8") as ftit:
        for idx, (src_idx, toks, lt, eos, gt) in enumerate(chosen):
            triplet_set = set(lt) | set(eos) | set(gt)
            lines = []
            cur = []
            for i, t in enumerate(toks):
                if i in triplet_set:
                    if i == lt[0] if lt else False:
                        pass
                    if t == ">":
                        lines.append(cur)
                        cur = []
                    continue
                cur.append(t)
            if cur:
                lines.append(cur)
            rec = {
                "subset_idx": idx,
                "src_idx": src_idx,
                "tokens": toks,
                "n_tokens": len(toks),
                "lt_positions": lt,
                "eos_positions": eos,
                "gt_positions": gt,
                "lines": lines,
            }
            fjson.write(json.dumps(rec, ensure_ascii=False) + "\n")
            tit = " ".join(lines[0][:5]) if lines else ""
            ftit.write(f"{idx}\t{src_idx}\t{tit}\t{len(toks)}\n")
            print(f"  [{idx}] src={src_idx}  L={len(toks)}  triplet@{lt[:2]}…{lt[-2:]}")

    print(f"\n[done] {out_json}\n       {out_titles}")


if __name__ == "__main__":
    main()
