"""M3-v1: 在 ci8w 上训练小型 BERT MLM，用于 LCVR 输出后的离散精炼。

词表与扩散模型一致（vocab.json），额外追加 [MASK] id = max_id + 1。
训练：随机 mask 字位（非标点、非 PAD），预测原 token。
不访问 HuggingFace Hub（from scratch 初始化）。

用法：
  cd improved-diffusion
  TRANSFORMERS_OFFLINE=1 python scripts/train_m3v1_mlm.py \\
      --vocab_dir diffusion_models/diff_e2e-tgt_block_2000steps_16dim_ci_sz12_gpu \\
      --corpus ../datasets/ci8w/ci_train.txt \\
      --out_dir diffusion_models/m3v1_mlm_ci8w \\
      --max_steps 50000 --batch_size 32 --save_every 5000
"""

from __future__ import annotations
import argparse
import functools
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "transformers" / "src"
if TRANSFORMERS_SRC.is_dir():
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import BertConfig, BertForMaskedLM  # noqa: E402

# 与 decode_pkl_lcvr 一致
PUNCT_CHARS = set("，。？！、；：「」（）—…")
SPECIAL_KEYS = {"START", "END", "PAD", "UNK", "STOP"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True, help="含 vocab.json 的目录（与 sz12 扩散一致）")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_len", type=int, default=144)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=50000)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--intermediate_size", type=int, default=2048)
    p.add_argument("--max_position_embeddings", type=int, default=256)
    return p.parse_args()


def load_vocab(vocab_dir: str):
    path = os.path.join(vocab_dir, "vocab.json")
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    unk_id = int(vocab["UNK"])
    pad_id = int(vocab["PAD"])
    max_id = max(int(v) for v in vocab.values())
    mask_id = max_id + 1
    vocab_size = max_id + 2
    punct_ids = {int(vocab[k]) for k in PUNCT_CHARS if k in vocab}
    special_ids = {int(vocab[k]) for k in SPECIAL_KEYS if k in vocab}
    return vocab, unk_id, pad_id, mask_id, vocab_size, punct_ids, special_ids


def encode_text(text: str, vocab: dict, unk_id: int, max_len: int, pad_id: int):
    ids = []
    for ch in text:
        if len(ids) >= max_len:
            break
        ids.append(int(vocab.get(ch, unk_id)))
    L = len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
    return ids, L


class CiMLMDataset(Dataset):
    def __init__(self, corpus_path: str, vocab: dict, unk_id: int, pad_id: int,
                 max_len: int, punct_ids: set, special_ids: set):
        self.samples = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("|", 1)
                text = parts[1] if len(parts) >= 2 else parts[0]
                if len(text) < 4:
                    continue
                ids, true_len = encode_text(text, vocab, unk_id, max_len, pad_id)
                self.samples.append((ids, true_len))
        self.vocab = vocab
        self.unk_id = unk_id
        self.pad_id = pad_id
        self.max_len = max_len
        self.punct_ids = punct_ids
        self.special_ids = special_ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, true_len = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), true_len


def collate_fn(batch, mask_id: int, pad_id: int, mask_prob: float,
               punct_ids: set, special_ids: set):
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    true_lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    B, T = input_ids.shape
    labels = torch.full_like(input_ids, -100)
    rand = torch.rand(B, T)
    for b in range(B):
        tl = int(true_lens[b].item())
        for t in range(tl):
            tid = int(input_ids[b, t].item())
            if tid == pad_id or tid in punct_ids or tid in special_ids:
                continue
            if rand[b, t] < mask_prob:
                labels[b, t] = tid
                input_ids[b, t] = mask_id
        if (labels[b] != -100).sum() == 0 and tl > 0:
            maskable = punct_ids | special_ids | {pad_id}
            candidates = [
                t for t in range(tl)
                if int(input_ids[b, t].item()) not in maskable
            ]
            if candidates:
                t = candidates[int(torch.randint(0, len(candidates), (1,)).item())]
                tid = int(input_ids[b, t].item())
                labels[b, t] = tid
                input_ids[b, t] = mask_id
    attention_mask = (input_ids != pad_id).long()
    return input_ids, attention_mask, labels


def linear_warmup_lr(step: int, warmup: int, base_lr: float, max_steps: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    return base_lr * max(0.0, (max_steps - step) / max(1.0, max_steps - warmup))


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab, unk_id, pad_id, mask_id, vocab_size, punct_ids, special_ids = load_vocab(
        args.vocab_dir
    )
    print(f"[info] vocab_size={vocab_size} (incl. MASK={mask_id}), pad_id={pad_id}")

    ds = CiMLMDataset(args.corpus, vocab, unk_id, pad_id, args.max_len, punct_ids, special_ids)
    print(f"[info] dataset size = {len(ds)}")

    collate = functools.partial(
        collate_fn,
        mask_id=mask_id,
        pad_id=pad_id,
        mask_prob=args.mask_prob,
        punct_ids=punct_ids,
        special_ids=special_ids,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        drop_last=True,
    )

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        type_vocab_size=1,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
    )
    model = BertForMaskedLM(config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "vocab_dir": os.path.abspath(args.vocab_dir),
        "mask_token_id": mask_id,
        "pad_token_id": pad_id,
        "unk_token_id": unk_id,
        "max_len": args.max_len,
        "mask_prob": args.mask_prob,
        "corpus": os.path.abspath(args.corpus),
        "bert_config": config.to_dict(),
    }
    with open(out_dir / "m3v1_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    config.save_pretrained(out_dir)

    step = 0
    running_loss = 0.0
    data_iter = iter(loader)

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        lr_now = linear_warmup_lr(step, args.warmup_steps, args.lr, float(args.max_steps))
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        model.train()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        running_loss += loss.item()
        step += 1

        if step % args.log_every == 0:
            avg = running_loss / args.log_every
            running_loss = 0.0
            print(f"step {step}/{args.max_steps}  loss={avg:.4f}  lr={lr_now:.2e}")

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"mlm_step_{step}.pt"
            torch.save(
                {"step": step, "model_state_dict": model.state_dict(), "meta": meta},
                ckpt_path,
            )
            print(f"[info] saved {ckpt_path}")

    torch.save(
        {"step": step, "model_state_dict": model.state_dict(), "meta": meta},
        out_dir / "mlm_final.pt",
    )
    print(f"[done] final -> {out_dir / 'mlm_final.pt'}")


if __name__ == "__main__":
    main()
