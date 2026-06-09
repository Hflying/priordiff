#!/usr/bin/env python
"""Train a word-level GPT-2 small on sonnet_train.txt.

Fairness constraints (vs Diffusion-LM sonnet + LCVR):
  - Vocab = sonnet diffusion vocab (2868 tokens, shared START/END/UNK/PAD/<>/eos)
  - Param budget ~6M (matches our 16-channel, 128-d Diffusion backbone)
  - Tokenization = spacy English tokenizer (same as improved_diffusion text_datasets)
  - Training on the same 2685 sonnet_train.txt lines
  - Plain next-token CE loss, AdamW, cosine LR, no tricks

Output: saves ar_ckpt_ep{N}.pt under out_dir with model + vocab.
"""

import argparse, json, os, time, math
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, GPT2LMHeadModel
from spacy.lang.en import English


class SonnetWordDataset(Dataset):
    def __init__(self, txt_path, vocab, max_len=180):
        self.vocab = vocab
        self.START = vocab['START']
        self.END = vocab['END']
        self.UNK = vocab['UNK']
        self.PAD = vocab['PAD']
        self.max_len = max_len

        # Spacy tokenizer — same as improved_diffusion/text_datasets.py line 153
        nlp = English()
        tokenizer = nlp.tokenizer

        self.samples = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Spacy-tokenize: splits `<eos>` into `<` + `eos` + `>` etc.
                toks = [t.text for t in tokenizer(line)]
                ids = [self.START] + [vocab.get(t, self.UNK) for t in toks] + [self.END]
                # Truncate / pad to max_len
                if len(ids) > max_len:
                    ids = ids[:max_len - 1] + [self.END]
                else:
                    ids = ids + [self.PAD] * (max_len - len(ids))
                self.samples.append(ids)
        print(f'[data] {len(self.samples)} sonnets, max_len={max_len}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(self.samples[idx], dtype=torch.long)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_file', default='../datasets/sonnet3355/sonnet_train.txt')
    p.add_argument('--vocab_path',
                   default='diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355/vocab.json')
    p.add_argument('--out_dir', default='ar_baseline/char_gpt2_sonnet')
    p.add_argument('--max_len', type=int, default=180)
    p.add_argument('--n_layer', type=int, default=6)
    p.add_argument('--n_embd', type=int, default=256)
    p.add_argument('--n_head', type=int, default=8)
    p.add_argument('--epochs', type=int, default=60)  # 2685 lines -> 2x epochs
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--warmup_steps', type=int, default=200)
    p.add_argument('--save_every', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    with open(args.vocab_path) as f:
        vocab = json.load(f)
    print(f'[vocab] size={len(vocab)}')

    ds = SonnetWordDataset(args.train_file, vocab, args.max_len)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=2, pin_memory=True, drop_last=True)

    cfg = GPT2Config(
        vocab_size=len(vocab),
        n_positions=args.max_len,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=vocab['START'],
        eos_token_id=vocab['END'],
    )
    model = GPT2LMHeadModel(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[model] {args.n_layer}L × {args.n_embd}d × {args.n_head}h '
          f'= {n_params/1e6:.2f}M params')

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.1)
    total_steps = len(dl) * args.epochs

    def lr_sched(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    pad_id = vocab['PAD']
    t0 = time.time()
    step = 0
    model.train()
    for epoch in range(args.epochs):
        ep_loss = 0.0
        ep_n = 0
        for batch in dl:
            batch = batch.cuda(non_blocking=True)
            labels = batch.clone()
            labels[labels == pad_id] = -100
            out = model(input_ids=batch, labels=labels)
            loss = out.loss

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for g in opt.param_groups:
                g['lr'] = args.lr * lr_sched(step)
            opt.step()
            step += 1

            ep_loss += loss.item() * batch.size(0)
            ep_n += batch.size(0)
        avg = ep_loss / max(1, ep_n)
        print(f'[epoch {epoch+1:2d}/{args.epochs}] loss={avg:.4f} '
              f'ppl={math.exp(avg):.1f} '
              f'lr={args.lr * lr_sched(step):.2e} '
              f't={time.time()-t0:.0f}s', flush=True)

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ck = os.path.join(args.out_dir, f'ar_ckpt_ep{epoch+1}.pt')
            torch.save({'model': model.state_dict(),
                        'config': cfg.to_dict(),
                        'vocab': vocab,
                        'epoch': epoch + 1,
                        'args': vars(args)}, ck)
            print(f'[saved] {ck}')

    print(f'[done] total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
