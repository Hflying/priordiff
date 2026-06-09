"""
独立解码脚本：把 infill.py 周期性写出的 sample_dict checkpoint (.pkl)
解码成「可读中文 + 候选列表」的 JSON。

用法：
  cd improved-diffusion
  python scripts/decode_pkl.py \
      --model_path diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_gpu/ema_0.9999_200000.pt \
      --pkl out_gen/control_tone_vowel_length/checkpoint_tone_vowel_length_ci.pkl \
      --out out_gen/control_tone_vowel_length/decoded_partial.json

输出格式与原 infill.py 写出的 final json 一致（每行一个 Python repr dict）：
  {('PING','ZE',...): ['START 月 落 乌 啼 ...', '...', ...]}

可选：
  --topk 1            取最佳 1 候选 (默认即原行为)
  --strip_special     输出时去掉 START / END / PAD / UNK
  --pretty_json       输出标准 JSON（key 转字符串），便于二次处理
  --device cpu|cuda   默认 cuda（找不到就回退 cpu）
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch as th

from improved_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)
from improved_diffusion.rounding import load_tokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True, help="同 infill.py 的 --model_path（.pt 文件）")
    p.add_argument("--pkl", required=True, help="checkpoint pkl 路径")
    p.add_argument("--out", required=True, help="解码后写出的文件路径")
    p.add_argument("--topk", type=int, default=1, help="每个位置取 topk 候选；默认 1")
    p.add_argument("--strip_special", action="store_true", help="去掉 START/END/PAD/UNK")
    p.add_argument("--pretty_json", action="store_true",
                   help="按标准 JSON 写出（{key_str: [候选1, 候选2, ...]}），便于程序处理")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    return p.parse_args()


def build_model(model_path, device):
    """复刻 infill.py 里的模型构建：读 training_args.json，调用 create_model_and_diffusion。"""
    cfg_path = os.path.join(os.path.dirname(model_path), "training_args.json")
    with open(cfg_path, "r") as f:
        train_args = json.load(f)

    # 用 model_and_diffusion_defaults 的 keys 过滤
    defaults = model_and_diffusion_defaults()
    init_kwargs = {k: train_args.get(k, v) for k, v in defaults.items()}

    model, _diffusion = create_model_and_diffusion(**init_kwargs)
    state = th.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, train_args


def decode_array(model, arr, tokenizer, device, topk=1, strip_special=False):
    """arr: ndarray (num_samples, seqlen, in_channel) → list[str]"""
    SPECIAL = {"START", "END", "PAD", "UNK"}
    x_t = th.tensor(arr, dtype=th.float32, device=device)
    with th.no_grad():
        logits = model.get_logits(x_t)  # (B, T, V)
    if topk == 1:
        ids = logits.argmax(-1)  # (B, T)
        results = []
        for seq in ids:
            toks = [tokenizer[int(t)] for t in seq]
            if strip_special:
                toks = [t for t in toks if t not in SPECIAL]
            results.append(" ".join(toks))
        return results
    else:
        topk_ids = th.topk(logits, k=topk, dim=-1).indices  # (B, T, K)
        results = []
        for b in range(topk_ids.size(0)):
            cand_lst = []
            for k in range(topk):
                toks = [tokenizer[int(t)] for t in topk_ids[b, :, k]]
                if strip_special:
                    toks = [t for t in toks if t not in SPECIAL]
                cand_lst.append(" ".join(toks))
            # 同一个样本的多 topk 候选拼一起，便于查阅
            results.append(cand_lst)
        return results


def main():
    args = parse_args()

    if args.device == "cuda" and not th.cuda.is_available():
        print("[warn] CUDA 不可用，自动回退 CPU", file=sys.stderr)
        args.device = "cpu"
    device = th.device(args.device)

    print(f"[info] 加载 pkl: {args.pkl}")
    with open(args.pkl, "rb") as f:
        payload = pickle.load(f)
    sample_dict = payload["sample_dict"]
    meta = payload.get("meta", {})
    print(f"[info] sample_dict 条数 = {len(sample_dict)}  meta = {meta}")

    print(f"[info] 加载 diffusion 模型: {args.model_path}")
    model, _train_args = build_model(args.model_path, device)
    tokenizer = load_tokenizer("e2e-tgt", "random", os.path.dirname(args.model_path))
    print(f"[info] vocab 大小: {len(tokenizer)}")

    n_total = len(sample_dict)
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)

    decoded = []  # list[(key_tuple, word_lst)]
    for i, (k, arr) in enumerate(sample_dict.items(), 1):
        word_lst = decode_array(
            model, arr, tokenizer, device,
            topk=args.topk, strip_special=args.strip_special,
        )
        decoded.append((k, word_lst))
        if i % 10 == 0 or i == n_total:
            print(f"[info] decoded {i}/{n_total}")

    if args.pretty_json:
        # key 转成字符串，方便 JSON 序列化
        out = {}
        for k, v in decoded:
            out[" ".join(map(str, k))] = v
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    else:
        # 与 infill.py 原始输出格式一致：每行一个 dict 的 repr
        with open(args.out, "w", encoding="utf-8") as f:
            for k, v in decoded:
                print({k: v}, file=f)
            print("", file=f)
    print(f"[done] 写出: {args.out}  共 {len(decoded)} 条")


if __name__ == "__main__":
    main()
