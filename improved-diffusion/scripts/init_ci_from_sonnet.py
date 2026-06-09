"""把 sonnet diffusion ckpt 的 backbone 迁移到 ci diffusion model 上。

策略：
  - **可迁移**（in_channel/hidden_size 同 → 直接 copy weights）：
      input_up_proj, time_embed, input_transformers (BertEncoder 12L/12H/768d),
      position_embeddings (256, 768), LayerNorm, output_down_proj
  - **必须 reinit**（vocab 不同）：
      word_embedding, lm_head（共享权重）

输入：
  - sonnet ckpt:  diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355/ema_0.9999_100000.pt
  - ci 模型尺寸 config（image_size=12, vocab_size=5048）

输出：
  - 新 ckpt:  diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_init_from_sonnet/ema_0.9999_init.pt
  - training_args.json（ci 风格）+ vocab.json（ci 风格）
  - random_emb.torch（重新 init）

后续：用 bash/ci_finetune_from_sonnet.sh 触发 finetune。
"""

from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch as th

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# 必须 reinit（vocab 绑定）
VOCAB_BOUND_KEYS = {"word_embedding.weight", "lm_head.weight", "lm_head.bias"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sonnet_ckpt", required=True,
                   help="path to sonnet ema ckpt, e.g. diffusion_models/.../ema_0.9999_100000.pt")
    p.add_argument("--ci_template_dir", required=True,
                   help="existing ci ckpt dir (used as template for vocab.json/random_emb/training_args)")
    p.add_argument("--out_dir", required=True,
                   help="output dir; will be created if missing")
    p.add_argument("--out_step", type=int, default=0,
                   help="step suffix in output filename (default 0 = init)")
    p.add_argument("--print_only", action="store_true",
                   help="只打印转换计划，不写文件")
    return p.parse_args()


def main():
    args = parse_args()
    sonnet_ckpt_path = Path(args.sonnet_ckpt)
    ci_dir = Path(args.ci_template_dir)
    out_dir = Path(args.out_dir)

    if not sonnet_ckpt_path.exists():
        raise FileNotFoundError(sonnet_ckpt_path)
    if not (ci_dir / "vocab.json").exists():
        raise FileNotFoundError(f"{ci_dir}/vocab.json")
    if not (ci_dir / "training_args.json").exists():
        raise FileNotFoundError(f"{ci_dir}/training_args.json")

    print(f"[info] loading sonnet ckpt: {sonnet_ckpt_path}")
    sonnet_state = th.load(sonnet_ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sonnet_state, dict) and "model" in sonnet_state:
        sonnet_state = sonnet_state["model"]
    print(f"[info] sonnet has {len(sonnet_state)} params")

    print(f"[info] loading ci template ckpt for shape reference")
    ci_template_ckpts = list(ci_dir.glob("ema_0.9999_*.pt"))
    if not ci_template_ckpts:
        raise FileNotFoundError(f"no ema ckpt in {ci_dir}")
    ci_template_ckpts.sort()
    ci_state = th.load(ci_template_ckpts[0], map_location="cpu", weights_only=False)
    if isinstance(ci_state, dict) and "model" in ci_state:
        ci_state = ci_state["model"]
    print(f"[info] ci template ({ci_template_ckpts[0].name}) has {len(ci_state)} params")

    # 配对每个 key，决定来源
    new_state = {}
    n_copied = 0
    n_reinit = 0
    n_skipped = 0
    skipped_keys = []
    shape_mismatch = []

    for key, ci_tensor in ci_state.items():
        if key in VOCAB_BOUND_KEYS:
            new_state[key] = ci_tensor.clone()  # 保留 ci 原 random init
            n_reinit += 1
            continue
        if key not in sonnet_state:
            new_state[key] = ci_tensor.clone()
            n_skipped += 1
            skipped_keys.append(key)
            continue
        sonnet_tensor = sonnet_state[key]
        if sonnet_tensor.shape != ci_tensor.shape:
            new_state[key] = ci_tensor.clone()
            shape_mismatch.append((key, tuple(sonnet_tensor.shape), tuple(ci_tensor.shape)))
            n_skipped += 1
            continue
        new_state[key] = sonnet_tensor.clone()
        n_copied += 1

    print(f"\n[plan] {n_copied} copied from sonnet; {n_reinit} reinit (vocab-bound); {n_skipped} skipped")
    if skipped_keys:
        print(f"[plan] skipped (not in sonnet): {skipped_keys[:10]}{'...' if len(skipped_keys)>10 else ''}")
    if shape_mismatch:
        print(f"[plan] shape mismatch (kept ci):")
        for k, s_shape, c_shape in shape_mismatch[:10]:
            print(f"    {k}: sonnet {s_shape} -> ci {c_shape}")

    if args.print_only:
        print("\n[print_only] no files written")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    # copy support files
    for fname in ["vocab.json", "training_args.json", "random_emb.torch"]:
        src = ci_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            print(f"[ok] copied {fname} from ci template")

    # write merged ckpt
    ema_out = out_dir / f"ema_0.9999_{args.out_step:06d}.pt"
    th.save(new_state, ema_out)
    # 同时存一份 model000000.pt 作为非-EMA 起点
    model_out = out_dir / f"model{args.out_step:06d}.pt"
    th.save(new_state, model_out)
    print(f"\n[done] wrote {ema_out}")
    print(f"       wrote {model_out}")
    print(f"       (training_args.json / vocab.json / random_emb.torch 已就位)")
    print(f"\n下一步: bash/ci_finetune_from_sonnet.sh 触发 finetune")


if __name__ == "__main__":
    main()
