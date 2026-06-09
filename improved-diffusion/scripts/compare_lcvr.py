"""比较 baseline (bfreeze) / LCVR no_freq / LCVR drop_low_zero 三种解码策略.

baseline: out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_step{step}k_bfreeze_subset10.json
lcvr:     out_gen/control_tone_vowel_length/infill_lcvr_{step}k_bfreeze_{mode}.json

均按 target_ci_subset10.json 顺序 1-1 对应 10 条 target.

输出指标（每条 target 的 50 个候选取平均，再按 ckpt × mode 聚合）：
  - head_special_rate: head 区出现特殊 token 的比例
  - head_punct_misalign: head 区位 i 应该是字（target words_[i] 是字）但生成是标点 的位置占比
  - head_zero_freq_rate: head 区字位生成的字训练频次 = 0 的占比
  - head_low_freq_rate: head 区字位生成的字训练频次 ≤ 31 (20% 分位) 的占比
  - head_common_rate: head 区生成 token 落在常用 3500 字的比例
  - head_uniq_ratio: head 区 unique 字 / head 总长
  - tail_special_rate: tail [L+1, T) 区出现特殊 token (非 END/PAD) 的比例

用法：
  cd improved-diffusion
  python scripts/compare_lcvr.py
"""

from __future__ import annotations
import ast
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_readability import COMMON_3500 as COMMON  # noqa: E402

PUNCT = set("，。？！、；：「」（）—…")
SPECIAL = {"START", "END", "PAD", "UNK", "STOP"}
NEUTRAL_TAIL = {"END", "PAD"}


def load_lines_as_dicts(path):
    """infill 输出是 'one python-repr dict per line'."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = ast.literal_eval(line)
            out.append(d)
    return out


def load_targets(path):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_corpus_freq(corpus_path, vocab):
    cnt = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("|", 1)
            text = parts[1] if len(parts) >= 2 else parts[0]
            for ch in text:
                cnt[ch] += 1
    f_low_thresh = 31  # consistent with our drop_low_zero quantile (20%)
    return cnt, f_low_thresh


def per_target_metrics(cand_strs, target_words, char_freq, f_low):
    """
    cand_strs: list of `str` ('tok tok tok ...') – 50 candidates per target
    target_words: list[str] of len L (chars + puncts)
    Returns: dict of metrics averaged over candidates.
    """
    L = len(target_words)
    is_punct_target = [w in PUNCT for w in target_words]

    head_special, head_n = 0, 0
    head_misalign, head_char_n = 0, 0
    head_zero_freq, head_low_freq, head_char_freq_n = 0, 0, 0
    head_common, head_common_n = 0, 0
    head_uniq_chars, head_uniq_n = 0, 0
    tail_special, tail_n = 0, 0

    for s in cand_strs:
        toks = s.split()
        # head 区
        h_chars_for_uniq = []
        for i in range(min(L, len(toks))):
            t = toks[i]
            head_n += 1
            if t in SPECIAL:
                head_special += 1
                continue
            if is_punct_target[i]:
                # target 此位是标点，不计入字位 misalign
                continue
            # target 此位是字
            head_char_n += 1
            if t in PUNCT:
                head_misalign += 1
                continue
            # 是个字
            head_char_freq_n += 1
            f = char_freq.get(t, 0)
            if f == 0:
                head_zero_freq += 1
            if f <= f_low:
                head_low_freq += 1
            head_common_n += 1
            if t in COMMON:
                head_common += 1
            h_chars_for_uniq.append(t)

        if h_chars_for_uniq:
            head_uniq_chars += len(set(h_chars_for_uniq))
            head_uniq_n += len(h_chars_for_uniq)

        # tail 区 (L+1, len(toks))（跳过 boundary index L）
        for i in range(L + 1, len(toks)):
            t = toks[i]
            tail_n += 1
            if t in SPECIAL and t not in NEUTRAL_TAIL:
                # START/UNK/STOP 在 tail 都不希望出现
                tail_special += 1
            elif t not in SPECIAL and t not in PUNCT:
                # tail 区出现「字」也不希望（虽然 bfreeze partial_mask=False，
                # tail 仍可能漂出字）
                tail_special += 1

    safe = lambda a, b: a / b if b else 0.0
    return {
        "head_special": safe(head_special, head_n),
        "head_punct_misalign": safe(head_misalign, head_char_n),
        "head_zero_freq": safe(head_zero_freq, head_char_freq_n),
        "head_low_freq": safe(head_low_freq, head_char_freq_n),
        "head_common": safe(head_common, head_common_n),
        "head_uniq_ratio": safe(head_uniq_chars, head_uniq_n),
        "tail_outsider": safe(tail_special, tail_n),
    }


def aggregate(per_target_dicts):
    keys = list(per_target_dicts[0].keys())
    return {k: float(np.mean([d[k] for d in per_target_dicts])) for k in keys}


def main():
    out_dir = ROOT / "out_gen/control_tone_vowel_length"
    targets = load_targets(ROOT / "control_gen/target_ci_subset10.json")
    print(f"[info] {len(targets)} targets")

    vocab_path = ROOT / "diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu/vocab.json"
    vocab = json.load(open(vocab_path))
    char_freq, f_low = load_corpus_freq(ROOT.parent / "datasets/ci8w/ci_train.txt", vocab)
    print(f"[info] f_low (20%-quantile) = {f_low}")

    files = {
        "400k_baseline":     out_dir / "infill_tone_vowel_length_sz12_subset10_step400k_bfreeze_subset10.json",
        "400k_lcvr":         out_dir / "infill_lcvr_400k_bfreeze_no_freq.json",
        "400k_lcvr_freq":    out_dir / "infill_lcvr_400k_bfreeze_drop_low_zero.json",
        "400k_m3v0_a8_i10":  out_dir / "infill_m3v0_a8.0_i10_400k.json",
        "400k_m3v1":         out_dir / "infill_m3v1_400k.json",
    }

    # 每个 file 的 target index → cand_strs 列表
    rows = {}
    for tag, path in files.items():
        if not path.exists():
            print(f"[warn] missing: {path}")
            continue
        items = load_lines_as_dicts(path)
        # baseline 的 dict 是 {tone_tuple: [str, str, ...]}
        # lcvr 的 dict 也是 {tone_tuple: [str, str, ...]}
        # 顺序 = target 顺序
        per_target = []
        for i, d in enumerate(items):
            cand = list(d.values())[0]
            tw = targets[i]["words_"]
            m = per_target_metrics(cand, tw, char_freq, f_low)
            per_target.append(m)
        agg = aggregate(per_target)
        rows[tag] = {"agg": agg, "per": per_target}

    # 打印对比表
    metric_names = ["head_special", "head_punct_misalign", "head_zero_freq",
                    "head_low_freq", "head_common", "head_uniq_ratio", "tail_outsider"]
    header = ["config"] + [f"{m}" for m in metric_names]

    print("\n=== aggregate (mean over 10 targets, each averaged over 50 cand) ===\n")
    col_w = 22
    print(" ".join([h.ljust(col_w) for h in header]))
    for tag in files.keys():
        if tag not in rows:
            continue
        agg = rows[tag]["agg"]
        line = [tag.ljust(col_w)]
        for m in metric_names:
            line.append(f"{agg[m]*100:6.2f}%".ljust(col_w))
        print(" ".join(line))

    # 写 markdown 报告
    md_path = ROOT / "control_gen/lcvr_comparison.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# LCVR (M2) vs Baseline (bfreeze) 解码策略对比\n")
    md.append("数据：subset10 × 50 候选，按 target 平均，再按 candidate 平均。\n\n")
    md.append("- baseline = bfreeze 推理 + 无约束 argmax 解码（即原版输出）\n")
    md.append("- lcvr     = bfreeze 推理 + 位置约束 argmax（删除 special/标点错位）\n")
    md.append("- lcvr_freq = lcvr + 字位投影只允许 freq > 31 (训练 20% 分位) 的字\n\n")
    md.append("## 关键指标（百分比，越低越好除了 common/uniq）\n\n")
    md.append("| config | head_special↓ | head_punct_misalign↓ | head_zero_freq↓ | head_low_freq↓ | head_common↑ | head_uniq_ratio↑ | tail_outsider↓ |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for tag in files.keys():
        if tag not in rows:
            continue
        agg = rows[tag]["agg"]
        md.append(f"| {tag} | "
                  f"{agg['head_special']*100:.2f}% | "
                  f"{agg['head_punct_misalign']*100:.2f}% | "
                  f"{agg['head_zero_freq']*100:.2f}% | "
                  f"{agg['head_low_freq']*100:.2f}% | "
                  f"{agg['head_common']*100:.2f}% | "
                  f"{agg['head_uniq_ratio']*100:.2f}% | "
                  f"{agg['tail_outsider']*100:.2f}% |\n")
    md.append("\n## 指标解读\n\n")
    md.append("- `head_special`：head 区出现 START/END/PAD/UNK/STOP 的比例 → LCVR 应严格为 0\n")
    md.append("- `head_punct_misalign`：head 区**字位**生成了标点的比例 → LCVR 应严格为 0\n")
    md.append("- `head_zero_freq`：head 区**字位**生成训练 0 频字的比例 → 反映 vocab projection drift；lcvr_freq 严格为 0\n")
    md.append("- `head_low_freq`：head 区**字位**生成训练 < 20% 分位 (freq ≤ 31) 字的比例\n")
    md.append("- `head_common`：head 区生成 token 落在常用 3500 字的比例（衡量「人话」程度）\n")
    md.append("- `head_uniq_ratio`：head 区 unique 字 / 字总数（衡量重复字漂移；越高越多样）\n")
    md.append("- `tail_outsider`：tail 区出现非 END/PAD 的比例（bfreeze 残留的 tail 漂移）\n")
    md_path.write_text("".join(md), encoding="utf-8")
    print(f"\n[done] markdown report written to {md_path}")


if __name__ == "__main__":
    main()
