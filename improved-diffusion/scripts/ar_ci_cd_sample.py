#!/usr/bin/env python
"""AR-LM + Constrained Decoding baseline for ci.

Mechanism: autoregressive char-by-char generation from a GPT-2 small (6.07M).
At each position i, we apply the *same* V_i mask used by LCVR (§5):
  - i < L, target[i] is char:  V_i = content chars (vocab - PUNCT - SPECIAL)
  - i < L, target[i] is punct: V_i = PUNCT set
  - i == L (boundary):         V_i = {END}
  - i > L (tail):              V_i = {END, PAD}

Then argmax over V_i (masking out all non-V_i logits with -inf).

This tests whether LCVR's claimed mechanism is just AR+CD in disguise.
If CD-AR matches or beats Diffusion+LCVR on head_uniq while also satisfying
head_spec=0, tail_out=0 (structurally mandated by V_i), then the LCVR
mechanism has no unique value; our §6.1 story collapses.

Output: same `ast.literal_eval`-compatible format as decode_pkl_lcvr.py,
consumable by compare_lcvr.py without modification.
"""
from __future__ import annotations
import argparse, json, os
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

SPECIAL = {"START", "END", "PAD", "UNK", "STOP"}
# Aligned with compare_lcvr.py: PUNCT = set("，。？！、；：「」（）—…")
PUNCT = set("，。？！、；：「」（）—…")


def build_Vi_masks(vocab, target_words, seq_len):
    """Return a list of allowed-id sets, one per position i in [0, seq_len).

    For positions i < L: depends on target_words[i]
    For i == L: {END}
    For i > L: {END, PAD}
    """
    L = len(target_words)
    tok2id = vocab
    special_ids = {tok2id[t] for t in SPECIAL if t in tok2id}
    punct_ids = {tok2id[p] for p in PUNCT if p in tok2id}
    # "content" = vocab minus special minus punct
    content_ids = set(tok2id.values()) - special_ids - punct_ids

    END_ID = tok2id["END"]
    PAD_ID = tok2id["PAD"]

    V = []
    for i in range(seq_len):
        if i < L:
            if target_words[i] in PUNCT:
                V.append(punct_ids)
            else:
                V.append(content_ids)
        elif i == L:
            V.append({END_ID})
        else:
            V.append({END_ID, PAD_ID})
    return V


def sample_one_constrained(model, vocab, target_words, seq_len, device,
                           temperature=1.0, top_k=0, seed=None):
    """Generate one candidate: autoregressive with V_i mask at each step."""
    if seed is not None:
        torch.manual_seed(seed)

    V_list = build_Vi_masks(vocab, target_words, seq_len)
    START_ID = vocab["START"]

    # Build per-position allowed-mask matrix once (for speed): [T, V]
    V_size = len(vocab)
    mask = torch.full((seq_len, V_size), float("-inf"), device=device)
    for i, allowed in enumerate(V_list):
        idxs = torch.tensor(sorted(allowed), device=device, dtype=torch.long)
        mask[i, idxs] = 0.0

    # Autoregressive loop; context = [START, x_0, x_1, ..., x_{i-1}]; we predict x_i
    ctx = [START_ID]
    with torch.no_grad():
        for i in range(seq_len):
            ids = torch.tensor([ctx], device=device, dtype=torch.long)
            out = model(ids)
            logits = out.logits[0, -1]  # [V]
            logits = logits / max(1e-8, temperature)
            logits = logits + mask[i]   # -inf outside V_i

            if top_k and top_k > 0:
                top_vals, _ = torch.topk(logits, k=top_k)
                thresh = top_vals[-1]
                logits[logits < thresh] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            # guard: if V_i set was too restrictive and everything got -inf
            # (shouldn't happen since V_i is always non-empty by construction),
            # fall back to uniform over V_i
            if not torch.isfinite(probs).any() or probs.sum().item() <= 0:
                probs = torch.zeros_like(probs)
                probs[list(V_list[i])] = 1.0 / max(1, len(V_list[i]))

            next_id = int(torch.multinomial(probs, num_samples=1).item())
            ctx.append(next_id)

    return ctx[1:]  # drop START


@torch.no_grad()
def sample_constrained_batched(model, vocab, target_words, seq_len, n_cands,
                               device, temperature=1.0, top_k=0, seed=0,
                               use_cd=True):
    """Generate n_cands candidates in parallel using KV-cache.

    If use_cd=False, sample without V_i mask (unconstrained AR).
    Returns: list of n_cands lists of token ids (drop START).
    """
    torch.manual_seed(seed)
    V_logits = model.config.vocab_size

    if use_cd:
        V_list = build_Vi_masks(vocab, target_words, seq_len)
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
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        logits = logits / max(1e-8, temperature)
        if mask_mat is not None:
            logits = logits + mask_mat[i].unsqueeze(0)
        if top_k and top_k > 0:
            top_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            thresh = top_vals[:, -1:].expand_as(logits)
            logits = torch.where(logits < thresh, torch.full_like(logits, float("-inf")), logits)
        probs = F.softmax(logits, dim=-1)
        if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            row_sum = probs.sum(dim=-1, keepdim=True)
            probs = torch.where(row_sum > 0, probs / row_sum,
                                torch.full_like(probs, 1.0 / probs.size(-1)))
        next_ids = torch.multinomial(probs, num_samples=1)
        generated.append(next_ids)
        ctx = next_ids
    out_ids = torch.cat(generated, dim=1).cpu().tolist()  # [n_cands, T]
    return out_ids


def ids_to_tokstr(ids, id2tok):
    return " ".join(id2tok[i] for i in ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--target_json", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num_samples", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=160)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed_base", type=int, default=42)
    args = p.parse_args()

    # Load ckpt
    print(f"[load] {args.ckpt}")
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    vocab = ck["vocab"]
    cfg = GPT2Config(**ck["config"])
    model = GPT2LMHeadModel(cfg).to(args.device)
    model.load_state_dict(ck["model"])
    model.eval()
    id2tok = {v: k for k, v in vocab.items()}
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    print(f"[vocab] {len(vocab)}")

    # Load targets
    targets = []
    with open(args.target_json) as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    print(f"[targets] {len(targets)} entries; num_samples={args.num_samples}")

    # Generate
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fout:
        for t_idx, tgt in enumerate(targets):
            words = tgt["words_"]
            key = tuple(tgt["tone"])
            candidates = []
            for s_idx in range(args.num_samples):
                ids = sample_one_constrained(
                    model, vocab, words, args.seq_len,
                    device=args.device,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    seed=args.seed_base + t_idx * 1000 + s_idx,
                )
                candidates.append(ids_to_tokstr(ids, id2tok))
            # Write one python-repr dict per line (matches compare_lcvr.load_lines_as_dicts)
            fout.write(repr({key: candidates}) + "\n")
            print(f"  target {t_idx+1}/{len(targets)} done")

    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
