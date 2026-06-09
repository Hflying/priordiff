"""E2E NLG 评测：baseline vs LCVR。

指标：
  - head_special  : i < L 范围内 START/END/UNK/PAD 比例（专测 special leak）
  - tail_outsider : i ≥ L 范围内非 PAD（且非 END@i==L）比例
  - bound_align   : i == L 是 END 的比例（model 是否在边界出 END）
  - head_uniq     : 词位中 unique word / total word
  - slot_fidelity : 8 个 slot value 在 utterance 中正确出现的比例
                    （value 的 token 序列以子串形式出现）
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPECIAL_BASE = {"START", "END", "UNK", "PAD"}


def has_subseq(seq, sub):
    """sub 是否作为连续子序列出现在 seq 中。"""
    if not sub:
        return False
    n, m = len(seq), len(sub)
    for i in range(n - m + 1):
        if seq[i:i + m] == sub:
            return True
    return False


def metrics_for_candidate(cand_tokens, target_rec):
    T = len(cand_tokens)
    L = target_rec["n_tokens"]
    head = cand_tokens[:L]
    tail = cand_tokens[L:]

    head_special = sum(1 for t in head if t in SPECIAL_BASE) / max(1, len(head))
    bound_align = 1.0 if (L < T and cand_tokens[L] == "END") else 0.0
    if len(tail) > 1:
        tail_pad_zone = tail[1:]
        tail_outsider = sum(1 for t in tail_pad_zone if t != "PAD") / max(1, len(tail_pad_zone))
    else:
        tail_outsider = 0.0

    word_toks = [t for t in head if t not in SPECIAL_BASE]
    head_uniq = len(set(word_toks)) / max(1, len(word_toks))

    # slot fidelity
    value_tokens = target_rec.get("value_tokens", [])
    if value_tokens:
        hit = sum(1 for _, vt in value_tokens if has_subseq(cand_tokens, vt))
        slot_fidelity = hit / len(value_tokens)
    else:
        slot_fidelity = 0.0

    return dict(
        head_special=head_special,
        bound_align=bound_align,
        tail_outsider=tail_outsider,
        head_uniq=head_uniq,
        slot_fidelity=slot_fidelity,
    )


def aggregate(per_target):
    if not per_target:
        return {}
    keys = list(per_target[0].keys())
    return {k: sum(d[k] for d in per_target) / len(per_target) for k in keys}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_json", required=True)
    p.add_argument("--target_jsonl", default="control_gen/target_e2e30.jsonl")
    p.add_argument("--out_md", default="control_gen/e2e_compare.md")
    p.add_argument("--out_dir", default="control_gen/decoded_e2e30")
    args = p.parse_args()

    out_json_path = Path(args.out_json)
    if not out_json_path.is_absolute():
        out_json_path = ROOT / out_json_path
    target_path = Path(args.target_jsonl)
    if not target_path.is_absolute():
        target_path = ROOT / target_path

    print(f"[info] reading {out_json_path}")
    with open(out_json_path, "r") as f:
        data = json.load(f)

    targets = []
    with open(target_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfgs = ["baseline", "lcvr"]
    if "m3v1" in data:
        cfgs.append("m3v1")
    per_target_avg = {c: [] for c in cfgs}
    for tgt in targets:
        idx = str(tgt["subset_idx"])
        baseline_cands = data["baseline"]
        lcvr_cands = data["lcvr"][idx]["tokens"]
        m3v1_cands = data["m3v1"][idx]["tokens"] if "m3v1" in data else None

        bm = [metrics_for_candidate(c, tgt) for c in baseline_cands]
        lm = [metrics_for_candidate(c, tgt) for c in lcvr_cands]
        per_target_avg["baseline"].append(aggregate(bm))
        per_target_avg["lcvr"].append(aggregate(lm))
        if m3v1_cands is not None:
            mm = [metrics_for_candidate(c, tgt) for c in m3v1_cands]
            per_target_avg["m3v1"].append(aggregate(mm))

        readable_path = out_dir / f"{int(idx):02d}_{tgt['slots'].get('name','?').replace(' ','_')}.txt"
        with open(readable_path, "w", encoding="utf-8") as f:
            f.write(f"== E2E {idx} (src={tgt['src_idx']}, L={tgt['n_tokens']}) ==\n")
            f.write(f"\nMR: {tgt['mr_str']}\n")
            f.write(f"\n--- 真值 ---\n{tgt['target_str']}\n")
            f.write("\n--- baseline cand 0 ---\n" + " ".join(baseline_cands[0]) + "\n")
            f.write("\n--- LCVR cand 0 ---\n" + " ".join(lcvr_cands[0]) + "\n")
            f.write("\n--- LCVR cand 1 ---\n" + " ".join(lcvr_cands[1] if len(lcvr_cands) > 1 else lcvr_cands[0]) + "\n")
            f.write("\n--- LCVR cand 2 ---\n" + " ".join(lcvr_cands[2] if len(lcvr_cands) > 2 else lcvr_cands[0]) + "\n")

    agg = {k: aggregate(per_target_avg[k]) for k in per_target_avg}

    print("\n=== aggregate (mean over 30 targets, each averaged over 50 cand) ===\n")
    keys = ["head_special", "tail_outsider", "bound_align", "head_uniq", "slot_fidelity"]
    print("config".ljust(20) + "".join(k.ljust(18) for k in keys))
    for cfg, m in agg.items():
        cells = [f"{m[k]*100:6.2f}%".ljust(18) for k in keys]
        print(cfg.ljust(20) + "".join(cells))

    out_md = Path(args.out_md)
    if not out_md.is_absolute():
        out_md = ROOT / out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# E2E NLG baseline vs LCVR\n"]
    lines.append("| config | head_special↓ | tail_outsider↓ | bound_align↑ | head_uniq↑ | slot_fidelity↑ |")
    lines.append("|---|---|---|---|---|---|")
    for cfg, m in agg.items():
        lines.append(
            f"| {cfg} | {m['head_special']*100:.2f}% | {m['tail_outsider']*100:.2f}% "
            f"| {m['bound_align']*100:.2f}% | {m['head_uniq']*100:.2f}% | {m['slot_fidelity']*100:.2f}% |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[ok] markdown -> {out_md}")
    print(f"[ok] readable -> {out_dir}/")


if __name__ == "__main__":
    main()
