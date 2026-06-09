"""M3-v1 MLM trainer for sonnet3355 — word-level analogue of
`train_m3v1_mlm.py`.

Differences from the ci version:

* Word-level tokenization (spaCy English) instead of character-level.
* Vocabulary aligned with the sonnet diffusion model
  (`diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355/vocab.json`,
  2868 tokens including START/END/PAD/UNK and the line-break triplet
  `<` / `eos` / `>`).
* Protected (never-masked) positions: the structural anchors
  `<`/`eos`/`>`, plus all specials, plus English punctuation tokens.
* `max_len` defaults to 196 to match image_size=14.
* Smaller default `max_steps` (~10k) since the corpus has only 2,685
  sonnets versus ci8w's 74k lines.

Usage:
  cd improved-diffusion
  TRANSFORMERS_OFFLINE=1 python scripts/train_m3v1_mlm_sonnet.py \\
      --vocab_dir diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355 \\
      --corpus ../datasets/sonnet3355/sonnet_train.txt \\
      --out_dir diffusion_models/m3v1_mlm_sonnet3355 \\
      --max_steps 10000 --batch_size 32 --save_every 2000
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

SPECIAL_KEYS = {"START", "END", "PAD", "UNK"}
TRIPLET_KEYS = {"<", "eos", ">"}
PUNCT_TOKENS = {",", ".", ";", ":", "!", "?", "'", "\"", "(", ")", "-",
                "—", "…", "''", "``", "`", "n't"}

_TOK = None


def tokenize_en(s: str):
    global _TOK
    if _TOK is None:
        from spacy.lang.en import English
        _TOK = English().tokenizer
    return [t.text for t in _TOK(s)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--max_len", type=int, default=196)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--mask_prob", type=float, default=0.15)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--num_attention_heads", type=int, default=8)
    p.add_argument("--intermediate_size", type=int, default=2048)
    p.add_argument("--max_position_embeddings", type=int, default=256)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
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
    triplet_ids = {int(vocab[k]) for k in TRIPLET_KEYS if k in vocab}
    special_ids = {int(vocab[k]) for k in SPECIAL_KEYS if k in vocab}
    punct_ids = {int(vocab[w]) for w in PUNCT_TOKENS if w in vocab}
    protected_ids = special_ids | triplet_ids | punct_ids
    return vocab, unk_id, pad_id, mask_id, vocab_size, protected_ids


def encode_text(text: str, vocab: dict, unk_id: int, max_len: int, pad_id: int):
    ids = []
    for tok in tokenize_en(text):
        if len(ids) >= max_len:
            break
        ids.append(int(vocab.get(tok, unk_id)))
    L = len(ids)
    while len(ids) < max_len:
        ids.append(pad_id)
    return ids, L


class SonnetMLMDataset(Dataset):
    def __init__(self, corpus_path: str, vocab: dict, unk_id: int, pad_id: int,
                 max_len: int, protected_ids: set):
        self.samples = []
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                ids, true_len = encode_text(line, vocab, unk_id, max_len, pad_id)
                if true_len < 8:
                    continue
                self.samples.append((ids, true_len))
        self.protected_ids = protected_ids
        self.pad_id = pad_id
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, true_len = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), true_len


def collate_fn(batch, mask_id: int, pad_id: int, mask_prob: float,
               protected_ids: set):
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    true_lens = torch.tensor([b[1] for b in batch], dtype=torch.long)
    B, T = input_ids.shape
    labels = torch.full_like(input_ids, -100)
    rand = torch.rand(B, T)
    for b in range(B):
        tl = int(true_lens[b].item())
        for t in range(tl):
            tid = int(input_ids[b, t].item())
            if tid == pad_id or tid in protected_ids:
                continue
            if rand[b, t] < mask_prob:
                labels[b, t] = tid
                input_ids[b, t] = mask_id
        if (labels[b] != -100).sum() == 0 and tl > 0:
            candidates = [
                t for t in range(tl)
                if int(input_ids[b, t].item()) not in protected_ids
                and int(input_ids[b, t].item()) != pad_id
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
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] device={device}")

    (vocab, unk_id, pad_id, mask_id, vocab_size,
     protected_ids) = load_vocab(args.vocab_dir)
    print(f"[info] vocab_size={vocab_size} (incl. MASK={mask_id}), "
          f"pad_id={pad_id}, |protected|={len(protected_ids)}")

    ds = SonnetMLMDataset(args.corpus, vocab, unk_id, pad_id, args.max_len,
                          protected_ids)
    print(f"[info] dataset size = {len(ds)}")

    collate = functools.partial(
        collate_fn,
        mask_id=mask_id,
        pad_id=pad_id,
        mask_prob=args.mask_prob,
        protected_ids=protected_ids,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
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
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
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
        "tokenizer": "spaCy English",
        "protected_ids_count": len(protected_ids),
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
        lr_now = linear_warmup_lr(step, args.warmup_steps, args.lr,
                                  float(args.max_steps))
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
            print(f"step {step}/{args.max_steps}  loss={avg:.4f}  "
                  f"lr={lr_now:.2e}", flush=True)

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_path = out_dir / f"mlm_step_{step}.pt"
            torch.save(
                {"step": step, "model_state_dict": model.state_dict(),
                 "meta": meta},
                ckpt_path,
            )
            print(f"[info] saved {ckpt_path}", flush=True)

    torch.save(
        {"step": step, "model_state_dict": model.state_dict(), "meta": meta},
        out_dir / "mlm_final.pt",
    )
    print(f"[done] final -> {out_dir / 'mlm_final.pt'}", flush=True)


if __name__ == "__main__":
    main()
