#!/usr/bin/env python
"""AR-LM + Constrained Decoding baseline for sonnet.

Mechanism: autoregressive word-by-word generation from a GPT-2 small (5.52M).
At each position i, we apply the *same* V_i mask used by LCVR (§5):
  - i ∈ lt_positions  → V_i = {"<"}
  - i ∈ eos_positions → V_i = {"eos"}
  - i ∈ gt_positions  → V_i = {">"}
  - i > max(gt_positions) (tail) → V_i = {PAD}
  - else (word slot)  → V_i = vocab \\ {START, END, UNK, PAD, "<", "eos", ">"}

Then sample from softmax restricted to V_i.

This tests whether LCVR's claimed mechanism is just AR+CD in disguise.
If CD-AR matches Diffusion+LCVR on head_uniq with the same structural perfection
(head_special=0, head_eos_misalign=0, tail_outsider=0 are mandated by V_i),
then LCVR's mechanism collapses to constrained decoding.

Output: JSON in the same shape as `sonnet_sample_lcvr.py` so `sonnet_eval.py`
consumes it directly:
  {
    "baseline": [[tok,...], ...]   # AR samples WITHOUT V_i mask (unconstrained AR)
    "lcvr"    : {idx_str: {"src_idx": int, "n_tokens": int,
                           "tokens": [[tok,...], ...]}}  # AR + CD
  }
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


SPECIAL_BASE = {"START", "END", "UNK", "PAD"}
LINE_TRIPLET = {"<", "eos", ">"}


def build_Vi_masks(vocab, target_rec, seq_len):
    """Return list of allowed-id sets, one per position i in [0, seq_len)."""
    lt_pos = set(target_rec["lt_positions"])
    eos_pos = set(target_rec["eos_positions"])
    gt_pos = set(target_rec["gt_positions"])
    last_triplet_end = max(target_rec["gt_positions"])

    lt_id = vocab["<"]
    eos_word_id = vocab["eos"]
    gt_id = vocab[">"]
    pad_id = vocab["PAD"]

    forbidden_in_word = {vocab[t] for t in (SPECIAL_BASE | LINE_TRIPLET) if t in vocab}
    content_ids = set(vocab.values()) - forbidden_in_word

    V = []
    for i in range(seq_len):
        if i in lt_pos:
            V.append({lt_id})
        elif i in eos_pos:
            V.append({eos_word_id})
        elif i in gt_pos:
            V.append({gt_id})
        elif i > last_triplet_end:
            V.append({pad_id})
        else:
            V.append(content_ids)
    return V


@torch.no_grad()
def sample_constrained_batched(model, vocab, target_rec, seq_len, n_cands,
                               device, temperature=1.0, top_k=0, seed=0,
                               use_cd=True):
    """Generate n_cands candidates in parallel using KV-cache.

    If use_cd=False, sample without V_i mask (unconstrained AR baseline).
    """
    torch.manual_seed(seed)
    V_size = len(vocab)
    V_logits = model.config.vocab_size

    if use_cd:
        V_list = build_Vi_masks(vocab, target_rec, seq_len)
        mask_mat = torch.full((seq_len, V_logits), float("-inf"), device=device)
        for i, allowed in enumerate(V_list):
            idxs = torch.tensor(sorted(allowed), device=device, dtype=torch.long)
            mask_mat[i, idxs] = 0.0
    else:
        mask_mat = None

    START_ID = vocab["START"]
    ctx = torch.full((n_cands, 1), START_ID, dtype=torch.long, device=device)
    past = None
    generated = []
    for i in range(seq_len):
        out = model(input_ids=ctx, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]              # [B, V]
        past = out.past_key_values
        logits = logits / max(1e-8, temperature)
        if mask_mat is not None:
            logits = logits + mask_mat[i].unsqueeze(0)
        if top_k and top_k > 0:
            top_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            thresh = top_vals[:, -1:].expand_as(logits)
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        # guard against degenerate rows
        if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
            # fallback: uniform over allowed (only triggers in unconstrained mode for nan-logits)
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            row_sum = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sum > 0, probs / row_sum,
                                torch.full_like(probs, 1.0 / probs.size(-1)))
        next_ids = torch.multinomial(probs, num_samples=1)  # [B, 1]
        generated.append(next_ids)
        ctx = next_ids

    return torch.cat(generated, dim=1).cpu()  # [B, T]


def ids_to_tokens(ids, inv_vocab):
    out = []
    for row in ids.tolist():
        out.append([inv_vocab.get(i, f"<{i}>") for i in row])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=180,
                   help="must match GPT2 n_positions (default 180 from trainer)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed_base", type=int, default=42)
    p.add_argument("--include_baseline", action="store_true",
                   help="also generate 50 unconstrained AR samples for "
                        "'baseline' key (target-independent)")
    args = p.parse_args()

    print(f"[load] {args.ckpt}", flush=True)
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    vocab = ck["vocab"]
    cfg = GPT2Config(**ck["config"])
    model = GPT2LMHeadModel(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params/1e6:.2f}M params, n_positions={cfg.n_positions}", flush=True)
    print(f"[vocab] |V| = {len(vocab)}", flush=True)

    assert args.seq_len <= cfg.n_positions, \
        f"seq_len {args.seq_len} > model n_positions {cfg.n_positions}"

    targets = []
    with open(args.target_json) as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    print(f"[targets] {len(targets)} entries; num_samples={args.num_samples}", flush=True)

    out = {"baseline": [], "lcvr": {}}

    if args.include_baseline:
        print("[baseline] generating unconstrained AR samples (target-independent)", flush=True)
        baseline_ids = sample_constrained_batched(
            model, vocab, targets[0], args.seq_len, args.num_samples,
            device=args.device, temperature=args.temperature, top_k=args.top_k,
            seed=args.seed_base, use_cd=False,
        )
        out["baseline"] = ids_to_tokens(baseline_ids, inv_vocab)
        print(f"  baseline cand0_head: {out['baseline'][0][:10]}", flush=True)

    for t_idx, tgt in enumerate(targets):
        idx = str(tgt["subset_idx"])
        ids = sample_constrained_batched(
            model, vocab, tgt, args.seq_len, args.num_samples,
            device=args.device, temperature=args.temperature, top_k=args.top_k,
            seed=args.seed_base + t_idx * 1000, use_cd=True,
        )
        toks = ids_to_tokens(ids, inv_vocab)
        out["lcvr"][idx] = {
            "src_idx": tgt["src_idx"],
            "n_tokens": tgt["n_tokens"],
            "tokens": toks,
        }
        print(f"  [{t_idx+1}/{len(targets)}] target {idx} done; cand0_head: {toks[0][:8]}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
