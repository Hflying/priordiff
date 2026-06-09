"""把 control_tone_vowel_length 的 infill JSON 输出格式化成可读对照文本。

输入：
  - infill JSON：     out_gen/control_tone_vowel_length/infill_*.json
                     格式：{(tone_tuple): [str, str, ...]}（每条约束 num_samples 个候选）
  - target subset：   control_gen/target_ci_subset10.json (10 行 jsonl)
  - target titles：   control_gen/target_ci_subset10_titles.txt
                     (subset_idx \t src_idx \t title \t length)

输出：
  - control_gen/decoded_subset10/<idx>_<title>.txt
       每词一文件，含：原词牌真值 + Top-N 候选（按按句切，过滤显眼噪声 token）
  - control_gen/decoded_subset10/_summary.md
       一份汇总报告：每个词牌列 1 个最佳候选 + 真值对照 + 简单统计
"""
import argparse
import ast
import json
import os
import re
from collections import Counter
from pathlib import Path

NOISE_TOKENS = {"■", "□", "UNK", "■■", "<unk>", "PAD"}
CONTROL_TOKENS = {"START", "END", "STOP"}
PUNCT = {"，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）"}

# 现代汉语常用字 ~1500（用于估算「常用字命中率」）
COMMON_CN = set("""的一是了我不人在他有这个上们来到时大地为子中你说生国年着
就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没
成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动
方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活
正感见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气每
并别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表德少克代员许
稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平报友
关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通
师立象数四失满战远格士音轻目条呢病始达深完今提求清王化空业思切怎非找片罗钱
紶吗语元喜曾离飞科言干流欢约各即指合反题必该论交终林请医晚制球决窢传画保读运
及则房早院量苦火布品近坐产答星精视五连司巴奇管类未朋且婚台夜青北队久乎越观
落尽形影红爸百令周吧识步希亚术留市半热送兴造谈容极随演收首根讲整式取照办强石
古华諣拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断福城故历惊脸选包紧争
另建维绝树系伤示愿持千史谁准联妇纪基买志关克令冷胜便英张景照集传朝调跟拌
""".replace("\n", ""))

ROOT = Path(__file__).resolve().parents[1]


def load_infill_dict(path: Path):
    """infill 写出的是逐行的 Python repr，每行一个 {tuple: [候选...]}."""
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = ast.literal_eval(line)
            for k, v in d.items():
                out[k] = v
    return out


def load_targets(json_path: Path, titles_path: Path):
    targets = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            targets.append(json.loads(line))
    titles = []
    with open(titles_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            titles.append({"subset_idx": int(parts[0]), "src_idx": int(parts[1]),
                            "title": parts[2], "length": int(parts[3]) if len(parts) >= 4 else 0})
    return targets, titles


def split_lines(toks):
    """按 ，。？！ 切句。每个 token 是空格分隔的字符。"""
    out = []
    cur = []
    PUNCT = {"，", "。", "？", "！"}
    for t in toks:
        if t in PUNCT:
            cur.append(t)
            out.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        out.append(cur)
    return out


def render_candidate(text: str, strip_noise: bool = True, head_len=None,
                      target_words=None, align_to_target: bool = False,
                      placeholder: str = "·"):
    """text 是空格分隔的字符序列。
    返回 (句子列表, 噪声率, 常用字命中率)。
    常用字命中率：内容字（去掉 ■/UNK/标点）中落在常用字表的比例。

    head_len: 若给定，只取 token 序列前 head_len 个位置（截断 boundary_freeze
    无法约束住的 [L, 144) 区，避免显示重复占位字尾巴）。

    align_to_target / target_words: 启用按真值标点位置切句模式。
        - 模型在「字位置」吐标点/控制符 → 用 placeholder 替换（不破坏句长）
        - 模型在「标点位置」未吐标点 → 强制用 target 该位置标点
    """
    toks = text.strip().split()
    if head_len is not None and head_len > 0:
        toks = toks[:head_len]
    total = len(toks)
    if total == 0:
        return [], 1.0, 0.0
    n_noise = sum(1 for t in toks if t in NOISE_TOKENS or t in CONTROL_TOKENS)
    content = [t for t in toks if t not in NOISE_TOKENS and t not in CONTROL_TOKENS and t not in PUNCT]
    if content:
        n_common = sum(1 for t in content if t in COMMON_CN)
        common_rate = n_common / len(content)
    else:
        common_rate = 0.0

    if align_to_target and target_words is not None:
        # 按 target 标点位置切句：保证每行字数和 target 一致
        L = min(len(toks), len(target_words))
        out_lines = []
        cur = []
        for i in range(L):
            t = toks[i]
            tw = target_words[i]
            if tw in PUNCT:
                # target 此位为标点
                cur.append(t if t in PUNCT else tw)
                out_lines.append("".join(cur))
                cur = []
            else:
                # target 此位为字
                if t in NOISE_TOKENS or t in CONTROL_TOKENS or t in PUNCT:
                    cur.append(placeholder)
                else:
                    cur.append(t)
        if cur:
            out_lines.append("".join(cur))
        return out_lines, n_noise / max(total, 1), common_rate

    if strip_noise:
        toks = [t for t in toks if t not in NOISE_TOKENS and t not in CONTROL_TOKENS]
    lines = split_lines(toks)
    rendered = ["".join(ln) for ln in lines]
    return rendered, n_noise / max(total, 1), common_rate


def render_target(words):
    """target 的 words_ 已经是字符列表（含标点）。"""
    PUNCT = {"，", "。", "？", "！"}
    out = []
    cur = []
    for w in words:
        cur.append(w)
        if w in PUNCT:
            out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--infill_json", default=str(ROOT / "out_gen/control_tone_vowel_length/infill_tone_vowel_length_sz12_subset10_subset10.json"))
    parser.add_argument("--target_json", default=str(ROOT / "control_gen/target_ci_subset10.json"))
    parser.add_argument("--titles", default=str(ROOT / "control_gen/target_ci_subset10_titles.txt"))
    parser.add_argument("--out_dir", default=str(ROOT / "control_gen/decoded_subset10"))
    parser.add_argument("--top_k", type=int, default=5, help="保留每个词牌噪声率最低的前 K 个候选")
    parser.add_argument("--truncate_to_target", action="store_true",
                        help="只显示前 L=len(target.words_) 个 token（去掉 boundary_freeze 无法约束住的 tail 区垃圾）")
    parser.add_argument("--align_to_target", action="store_true",
                        help="按 target 标点位置切句：保证每行字数对齐 target；模型在字位置吐的标点/控制符替换为 ·")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets, titles = load_targets(Path(args.target_json), Path(args.titles))
    infill = load_infill_dict(Path(args.infill_json))

    # infill 的 key 是 tone tuple；以 target 的 tone 序列匹配
    keys = list(infill.keys())
    print(f"[info] infill keys: {len(keys)}, targets: {len(targets)}")

    # 按 tone 内容做匹配（tone 序列在 _SEQ_LEN=144 长度下做 PAD，但 infill 里的 key 长度是？）
    # 先看一个 key 长度
    print(f"[info] sample key length: {len(keys[0])}, target[0] tone len: {len(targets[0]['tone'])}")

    # tone 在 infill 里是被 pad 到 SEQ_LEN 的；target 里没 pad。需要按前缀匹配。
    def find_key_for(target_tone):
        for k in keys:
            if list(k[: len(target_tone)]) == list(target_tone):
                return k
        return None

    summary_lines = ["# 生成结果对照（sz12 模型，subset10）", ""]
    summary_lines.append("候选按「常用字命中率」降序，越高越接近人话。")
    summary_lines.append("")
    summary_lines.append("| # | 词牌 | 字数 | best常用字% | avg常用字% | best噪声% |")
    summary_lines.append("|---|------|------|-----------|----------|----------|")

    for ti in titles:
        idx = ti["subset_idx"]
        title = ti["title"]
        src_idx = ti["src_idx"]
        target = targets[idx]

        key = find_key_for(target["tone"])
        if key is None:
            print(f"[warn] no match for {idx} {title}")
            continue

        candidates = infill[key]
        results = []
        head_len = len(target["words_"]) if (args.truncate_to_target or args.align_to_target) else None
        for cand_text in candidates:
            lines, noise, common = render_candidate(
                cand_text, strip_noise=True, head_len=head_len,
                target_words=target["words_"] if args.align_to_target else None,
                align_to_target=args.align_to_target,
            )
            results.append({"text": cand_text, "lines": lines, "noise": noise, "common": common})
        # 改成按「常用字命中率」降序排序，更能反映可读性
        results.sort(key=lambda r: (-r["common"], r["noise"]))

        best = results[0]
        avg_common = sum(r["common"] for r in results) / len(results)
        summary_lines.append(
            f"| {idx} | {title} | {len(target['words_'])} | "
            f"{best['common']:.2%} | {avg_common:.2%} | {best['noise']:.2%} |"
        )

        out_file = out_dir / f"{idx:02d}_{title}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"== 词牌 {idx} ({title})  | src_idx={src_idx} | 字数={len(target['words_'])} ==\n\n")
            f.write("--- 真值（target_ci.json）---\n")
            for ln in render_target(target["words_"]):
                f.write(ln + "\n")
            f.write("\n--- 模型生成 Top-{} 候选（按「常用字命中率」降序，去除 ■/UNK/□/START/END） ---\n\n".format(args.top_k))
            for j, r in enumerate(results[: args.top_k]):
                f.write(f"### 候选 {j+1}  (常用字={r['common']:.1%}, 噪声率={r['noise']:.1%})\n")
                for ln in r["lines"]:
                    f.write(ln + "\n")
                f.write("\n")
            f.write("\n--- 全部 50 个候选噪声率分布 ---\n")
            noise_buckets = Counter()
            for r in results:
                bucket = f"{int(r['noise']*10)*10}–{int(r['noise']*10)*10+10}%"
                noise_buckets[bucket] += 1
            for k in sorted(noise_buckets.keys()):
                f.write(f"  {k}: {noise_buckets[k]}\n")
        print(f"[ok] {out_file}  best_noise={best['noise']:.1%}")

    with open(out_dir / "_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
    print(f"\n[ok] summary -> {out_dir / '_summary.md'}")


if __name__ == "__main__":
    main()
