"""分析三个 ckpt 输出的「可读性」对比：200k / 300k / 400k。

指标：
  1. noise_rate   : ■/□/UNK/START/END 占比（之前的指标）
  2. common_rate  : 输出字符落在「常用 3500 汉字」表的比率（衡量「人话」程度）
  3. dup_run_max  : 最长重复 token 串（"在 在 在 在" 这种）
  4. uniq_ratio   : 不重复字符 / 总字符（多样性）

输入：
  - infill_*.json 三份（每份是一行一个 dict）
输出：
  - control_gen/decoded_subset10/_compare.md
"""
import argparse
import ast
import re
from collections import Counter
from pathlib import Path

# 现代汉语常用字 3500（截取自《现代汉语常用字表》一级字 3500，公开数据）。
# 这里用 GB2312 一级 3755 字近似，足以判别「人话」程度。
COMMON_3500 = """的一是了我不人在他有这个上们来到时大地为子中你说生国年着
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
""".replace("\n", "")

NOISE_TOKENS = {"■", "□", "UNK"}
CONTROL_TOKENS = {"START", "END", "STOP"}
PUNCT = {"，", "。", "？", "！", "、", "；", "：", "「", "」", "（", "）"}


def load_infill(path: Path):
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


def metrics(text: str):
    toks = text.strip().split()
    n = len(toks)
    if n == 0:
        return None

    n_noise = sum(1 for t in toks if t in NOISE_TOKENS or t in CONTROL_TOKENS)
    content = [t for t in toks if t not in NOISE_TOKENS and t not in CONTROL_TOKENS and t not in PUNCT]
    n_content = len(content)
    if n_content == 0:
        return None

    n_common = sum(1 for t in content if t in COMMON_3500)
    common_rate = n_common / n_content

    # 最长重复 run
    max_run = 1
    cur_run = 1
    for i in range(1, len(content)):
        if content[i] == content[i-1]:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 1

    uniq = len(set(content)) / max(n_content, 1)

    return {
        "noise_rate": n_noise / n,
        "common_rate": common_rate,
        "dup_run_max": max_run,
        "uniq_ratio": uniq,
        "n_content": n_content,
    }


def aggregate(infill_dict):
    """对一份输出，计算所有候选指标的均值 / best 候选指标。"""
    all_m = []
    best_per_task = []  # 每个 task 取常用字比率最高的候选
    for k, cands in infill_dict.items():
        task_m = []
        for c in cands:
            m = metrics(c)
            if m:
                task_m.append(m)
                all_m.append(m)
        if task_m:
            best_per_task.append(max(task_m, key=lambda x: x["common_rate"]))
    return all_m, best_per_task


def avg(lst, key):
    if not lst:
        return float("nan")
    return sum(x[key] for x in lst) / len(lst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="多个 (label, path) 对，例如 200k=path1 300k=path2")
    parser.add_argument("--out", default="control_gen/decoded_subset10/_compare.md")
    args = parser.parse_args()

    rows = []
    for spec in args.inputs:
        label, path = spec.split("=", 1)
        d = load_infill(Path(path))
        all_m, best_m = aggregate(d)
        rows.append({
            "label": label,
            "n_cand": len(all_m),
            "noise_avg": avg(all_m, "noise_rate"),
            "common_avg": avg(all_m, "common_rate"),
            "common_best_avg": avg(best_m, "common_rate"),
            "dup_avg": avg(all_m, "dup_run_max"),
            "uniq_avg": avg(all_m, "uniq_ratio"),
        })

    print(f"{'ckpt':<10}{'n':<6}{'noise%':<8}{'common%':<10}{'best_common%':<14}{'dup_max':<10}{'uniq%':<8}")
    print("-" * 70)
    for r in rows:
        print(f"{r['label']:<10}{r['n_cand']:<6}"
              f"{r['noise_avg']*100:<8.2f}"
              f"{r['common_avg']*100:<10.2f}"
              f"{r['common_best_avg']*100:<14.2f}"
              f"{r['dup_avg']:<10.2f}"
              f"{r['uniq_avg']*100:<8.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 不同 ckpt 输出的可读性对比\n\n")
        f.write("**指标说明**\n")
        f.write("- `noise%`：占位符 `■/□/UNK/START/END` 占比（越低越好）\n")
        f.write("- `common%`：内容字落在「现代汉语常用 ~1500 字表」的比率（**越高越接近人话**）\n")
        f.write("- `best_common%`：每首词牌取「常用字比率最高」的候选，再求均值\n")
        f.write("- `dup_max`：候选内最长重复 token 串（如 `在在在在` = 4，越低越好）\n")
        f.write("- `uniq%`：候选内不重复字符比率（越高越好，但不绝对，叠字如「风风」也合法）\n\n")
        f.write(f"| ckpt | n_cand | noise% | common% | best_common% | dup_max | uniq% |\n")
        f.write(f"|------|--------|--------|---------|-------------|--------|-------|\n")
        for r in rows:
            f.write(f"| {r['label']} | {r['n_cand']} "
                    f"| {r['noise_avg']*100:.2f} "
                    f"| {r['common_avg']*100:.2f} "
                    f"| {r['common_best_avg']*100:.2f} "
                    f"| {r['dup_avg']:.2f} "
                    f"| {r['uniq_avg']*100:.2f} |\n")
    print(f"\n[ok] -> {out_path}")


if __name__ == "__main__":
    main()
