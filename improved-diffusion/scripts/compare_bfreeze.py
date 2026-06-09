"""比较 baseline vs boundary-freeze 推理的可读性指标。

每条记录看：
  - head 区 [0, L)：noise_rate / common_rate / uniq_ratio
  - tail 区 [L, SEQ_LEN)：同上
  - 整体：noise_rate / common_rate / uniq_ratio / dup_run_max

通过对比可以验证：boundary_freeze 是否（1）显著降低 tail 区噪声，（2）不破坏 head 区质量。

用法：
  python scripts/compare_bfreeze.py \
      --baseline out_gen/control_tone_vowel_length/infill_..._step400k_subset10.json \
      --bfreeze  out_gen/control_tone_vowel_length/infill_..._step400k_bfreeze_subset10.json \
      --target   control_gen/target_ci_subset10.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

NOISE = {"□", "■", "UNK", "START", "END", "PAD", "STOP"}
PUNCT = {"，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）", "—"}

# 复用 analyze_readability.py 里的常用字表
sys.path.insert(0, str(Path(__file__).parent))
from analyze_readability import COMMON_3500 as COMMON  # noqa: E402


def load_infill(path: Path) -> list[tuple[tuple, list[str]]]:
    out: list[tuple[tuple, list[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = ast.literal_eval(line)
            for k, v in d.items():
                out.append((k, v))
    return out


def load_targets(path: Path) -> list[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def metrics_for_region(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {"noise": 0.0, "common": 0.0, "uniq": 0.0, "n": 0}
    n = len(tokens)
    noise = sum(1 for t in tokens if t in NOISE)
    content = [t for t in tokens if t not in NOISE and t not in PUNCT]
    # common_rate 用 n_common / n_total（含 noise/punct），让 noise 多时 common 自然低
    common = sum(1 for t in tokens if t in COMMON)
    uniq = len(set(content)) / max(1, len(content))
    return {
        "noise": noise / n,
        "common": common / n,
        "uniq": uniq,
        "n": n,
    }


def aggregate(records, targets, seq_len: int = 144) -> dict:
    """对所有词牌、所有候选汇总 head/tail/全文 三个区域的指标。"""
    head_acc = Counter()
    tail_acc = Counter()
    full_acc = Counter()
    n_head = n_tail = n_full = 0
    per_target = []

    for (tone_key, samples), tgt in zip(records, targets):
        L = len(tgt["words_"])
        L = min(L, seq_len)
        h_acc = Counter()
        t_acc = Counter()
        f_acc = Counter()
        nh = nt = nf = 0

        for s in samples:
            toks = s.split(" ")
            if len(toks) < seq_len:
                continue
            head_toks = toks[:L]
            tail_toks = toks[L:seq_len]
            full_toks = toks[:seq_len]

            mh = metrics_for_region(head_toks)
            mt = metrics_for_region(tail_toks)
            mf = metrics_for_region(full_toks)

            h_acc["noise"] += mh["noise"]
            h_acc["common"] += mh["common"]
            h_acc["uniq"] += mh["uniq"]
            t_acc["noise"] += mt["noise"]
            t_acc["common"] += mt["common"]
            t_acc["uniq"] += mt["uniq"]
            f_acc["noise"] += mf["noise"]
            f_acc["common"] += mf["common"]
            f_acc["uniq"] += mf["uniq"]
            nh += 1
            nt += 1
            nf += 1

        per_target.append({
            "L": L,
            "head_noise": h_acc["noise"] / max(1, nh),
            "tail_noise": t_acc["noise"] / max(1, nt),
            "head_common": h_acc["common"] / max(1, nh),
            "tail_common": t_acc["common"] / max(1, nt),
            "head_uniq": h_acc["uniq"] / max(1, nh),
            "tail_uniq": t_acc["uniq"] / max(1, nt),
        })
        for k, v in h_acc.items():
            head_acc[k] += v
        for k, v in t_acc.items():
            tail_acc[k] += v
        for k, v in f_acc.items():
            full_acc[k] += v
        n_head += nh
        n_tail += nt
        n_full += nf

    avg = {
        "head_noise": head_acc["noise"] / max(1, n_head),
        "tail_noise": tail_acc["noise"] / max(1, n_tail),
        "full_noise": full_acc["noise"] / max(1, n_full),
        "head_common": head_acc["common"] / max(1, n_head),
        "tail_common": tail_acc["common"] / max(1, n_tail),
        "full_common": full_acc["common"] / max(1, n_full),
        "head_uniq": head_acc["uniq"] / max(1, n_head),
        "tail_uniq": tail_acc["uniq"] / max(1, n_tail),
        "full_uniq": full_acc["uniq"] / max(1, n_full),
        "n": n_full,
    }
    return {"avg": avg, "per_target": per_target}


def fmt_pct(x: float) -> str:
    return f"{x*100:5.1f}%"


def print_compare(a: dict, b: dict, name_a: str, name_b: str):
    avg_a = a["avg"]
    avg_b = b["avg"]
    print()
    print(f"== Aggregate over {avg_a['n']}/{avg_b['n']} candidates ==")
    print("注：boundary_freeze 后 tail 区被强制为 END（NOISE），故 tail_noise→100%、tail_common→0% 是符合预期的，关键看 head 区。")
    print()
    print(f"{'metric':18}  {name_a:>12}  {name_b:>12}  {'Δ':>10}")
    print("-" * 60)
    for k in [
        "head_noise", "head_common", "head_uniq",
        "tail_noise", "tail_common", "tail_uniq",
        "full_noise", "full_common", "full_uniq",
    ]:
        va, vb = avg_a[k], avg_b[k]
        d = vb - va
        marker = ""
        if k == "head_noise":
            marker = "  <- key (越低越好)"
        elif k == "head_common":
            marker = "  <- key (越高越好)"
        print(f"{k:18}  {fmt_pct(va):>12}  {fmt_pct(vb):>12}  {d*100:+9.2f}pp{marker}")

    print()
    print(f"== Per-target ({len(a['per_target'])} items) — head 区指标对比 ==")
    print(f"{'#':>2}  {'L':>3}    {'head_noise':>17}    {'head_common':>17}    {'head_uniq':>17}")
    print(f"{'':2}     {'':3}    {name_a:>7}/{name_b:>7}    {name_a:>7}/{name_b:>7}    {name_a:>7}/{name_b:>7}")
    print("-" * 95)
    for i, (pa, pb) in enumerate(zip(a["per_target"], b["per_target"])):
        h_dn = (pb["head_noise"] - pa["head_noise"]) * 100
        h_dc = (pb["head_common"] - pa["head_common"]) * 100
        print(f"{i:>2}  {pa['L']:>3}    "
              f"{fmt_pct(pa['head_noise']):>7}/{fmt_pct(pb['head_noise']):>7} ({h_dn:+5.1f})  "
              f"{fmt_pct(pa['head_common']):>7}/{fmt_pct(pb['head_common']):>7} ({h_dc:+5.1f})  "
              f"{fmt_pct(pa['head_uniq']):>7}/{fmt_pct(pb['head_uniq']):>7}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--bfreeze", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--seq_len", type=int, default=144)
    args = p.parse_args()

    targets = load_targets(Path(args.target))
    base = load_infill(Path(args.baseline))
    bfre = load_infill(Path(args.bfreeze))

    n = min(len(base), len(bfre), len(targets))
    if n < len(targets):
        print(f"[warn] truncating to first {n} targets (baseline={len(base)}, bfreeze={len(bfre)}, "
              f"targets={len(targets)})", file=sys.stderr)
    a = aggregate(base[:n], targets[:n], args.seq_len)
    b = aggregate(bfre[:n], targets[:n], args.seq_len)
    print_compare(a, b, "baseline", "bfreeze")


if __name__ == "__main__":
    main()
