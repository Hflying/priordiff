"""E2E NLG-side M3-v1 refinement (parallel to ``decode_sonnet_m3_v1.py``).

E2E LCVR mask schema is the simplest of the three datasets:
  - i < L  : V_i = vocab \\ {START, END, UNK, PAD}
  - i == L : V_i = {END}
  - i > L  : V_i = {PAD}

There are no triplet anchors, so the protected positions are simply
the boundary (i == L) and tail (i > L); the head positions [0, L)
are all eligible for masking-and-refilling.

Usage:
  cd improved-diffusion
  TRANSFORMERS_OFFLINE=1 python scripts/decode_e2e_m3_v1.py \\
      --lcvr_json out_gen/e2e_e2e30_25000.json \\
      --target_jsonl control_gen/target_e2e30.jsonl \\
      --mlm_ckpt diffusion_models/m3v1_mlm_e2e/mlm_final.pt \\
      --vocab_dir diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_e2e \\
      --out out_gen/e2e_e2e30_25000_m3v1.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import torch as th

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "transformers" / "src"
if TRANSFORMERS_SRC.is_dir():
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import BertConfig, BertForMaskedLM  # noqa: E402

SPECIAL_BASE = {"START", "END", "UNK", "PAD"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lcvr_json", required=True)
    p.add_argument("--target_jsonl", required=True)
    p.add_argument("--mlm_ckpt", required=True)
    p.add_argument("--vocab_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--refine_iters", type=int, default=8)
    p.add_argument("--repeat_threshold", type=int, default=3)
    return p.parse_args()


def load_mlm(ckpt_path: str, device: th.device):
    ckpt = th.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta") or {}
    out_dir = Path(ckpt_path).parent
    if not (out_dir / "config.json").exists():
        raise FileNotFoundError(f"expect config.json beside {ckpt_path}")
    config = BertConfig.from_pretrained(str(out_dir))
    model = BertForMaskedLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    mask_id = int(meta.get("mask_token_id", config.vocab_size - 1))
    pad_id = int(meta.get("pad_token_id", 3))
    return model, mask_id, pad_id, config.vocab_size


def build_mlm_position_mask(L: int, T: int, vocab: dict, V_mlm: int):
    """E2E LCVR mask, sized to the MLM vocabulary (incl. MASK)."""
    V_real = len(vocab)
    mask = th.zeros(T, V_mlm, dtype=th.bool)
    pad_id = vocab["PAD"]
    end_id = vocab.get("END")
    forbidden = {vocab[t] for t in SPECIAL_BASE if t in vocab}
    word_row = th.zeros(V_mlm, dtype=th.bool)
    upper = min(V_real, V_mlm)
    word_row[:upper] = True
    for tid in forbidden:
        if tid < V_mlm:
            word_row[tid] = False
    for i in range(T):
        if i < L:
            mask[i] = word_row
        elif i == L and end_id is not None and end_id < V_mlm:
            mask[i, end_id] = True
        elif pad_id < V_mlm:
            mask[i, pad_id] = True
    return mask


def tokens_to_ids(token_list, vocab, unk_id, pad_id, T):
    ids = []
    for tok in token_list[:T]:
        ids.append(int(vocab.get(tok, unk_id)))
    while len(ids) < T:
        ids.append(pad_id)
    return th.tensor(ids, dtype=th.long)


def refine_batch(
    ids: th.Tensor,
    L: int,
    pmask: th.Tensor,
    mlm: BertForMaskedLM,
    mask_id: int,
    pad_id: int,
    repeat_threshold: int,
    max_iters: int,
    device: th.device,
):
    """ids: (B, T). word_positions = [0, L)."""
    B, T = ids.shape
    word_positions = list(range(min(L, T)))
    if not word_positions:
        return ids
    Vlog = mlm.config.vocab_size
    pmask = pmask.to(device)
    if pmask.size(1) != Vlog:
        raise ValueError(
            f"pmask vocab dim {pmask.size(1)} != MLM vocab_size {Vlog}"
        )

    ids = ids.clone().to(device)
    word_idx = th.tensor(word_positions, device=device, dtype=th.long)
    pm_word = pmask.index_select(0, word_idx).unsqueeze(0)  # (1, |W|, V_mlm)

    for _ in range(max_iters):
        ids_word = ids.index_select(1, word_idx)
        cnt_batch = th.zeros(B, Vlog, device=device, dtype=th.float)
        cnt_batch.scatter_add_(
            1, ids_word, th.ones_like(ids_word, dtype=th.float)
        )
        cnt_per_pos = cnt_batch.gather(1, ids_word)
        to_fix_word = cnt_per_pos >= float(repeat_threshold)
        if not to_fix_word.any():
            break

        inp = ids.clone()
        inp_word = inp.index_select(1, word_idx)
        inp_word = th.where(
            to_fix_word, th.full_like(inp_word, mask_id), inp_word
        )
        inp.index_copy_(1, word_idx, inp_word)
        attn = (inp != pad_id).long()
        with th.no_grad():
            logits = mlm(input_ids=inp, attention_mask=attn).logits
        logits_word = logits.index_select(1, word_idx)
        logits_word = logits_word.masked_fill(~pm_word, float("-inf"))
        new_ids_word = logits_word.argmax(dim=-1)
        ids_word_new = th.where(to_fix_word, new_ids_word, ids_word)
        ids.index_copy_(1, word_idx, ids_word_new)
    return ids.cpu()


def main():
    args = parse_args()
    device = th.device(args.device if th.cuda.is_available() else "cpu")
    print(f"[info] device={device}", flush=True)

    vocab = json.load(open(os.path.join(args.vocab_dir, "vocab.json")))
    inv_vocab = {int(v): k for k, v in vocab.items()}
    unk_id = int(vocab["UNK"])
    pad_id = int(vocab["PAD"])

    print(f"[info] loading MLM from {args.mlm_ckpt}", flush=True)
    mlm, mask_id, _meta_pad, V_mlm = load_mlm(args.mlm_ckpt, device)
    print(f"[info] MLM vocab_size={V_mlm} mask_id={mask_id}", flush=True)

    targets = []
    with open(args.target_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                targets.append(json.loads(line))
    print(f"[info] {len(targets)} targets", flush=True)

    with open(args.lcvr_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "lcvr" not in data:
        raise SystemExit("--lcvr_json missing 'lcvr' key")
    print(f"[info] LCVR has {len(data['lcvr'])} records", flush=True)

    out_data = {"baseline": data.get("baseline", []),
                "lcvr": data["lcvr"], "m3v1": {}}
    for tgt in targets:
        idx = str(tgt["subset_idx"])
        cands_tokens = data["lcvr"][idx]["tokens"]
        T = len(cands_tokens[0])
        L = int(tgt["n_tokens"])
        rows = [tokens_to_ids(c, vocab, unk_id, pad_id, T) for c in cands_tokens]
        ids = th.stack(rows, dim=0)
        pmask = build_mlm_position_mask(L, T, vocab, V_mlm)
        refined = refine_batch(
            ids, L, pmask, mlm, mask_id, pad_id,
            args.repeat_threshold, args.refine_iters, device,
        )
        new_tokens = []
        for row in refined.tolist():
            new_tokens.append([inv_vocab.get(t, f"<{t}>") for t in row])
        out_data["m3v1"][idx] = {
            "src_idx": tgt["src_idx"],
            "n_tokens": L,
            "slots": tgt.get("slots", {}),
            "value_tokens": tgt.get("value_tokens", []),
            "tokens": new_tokens,
        }
        print(f"  [{idx}] L={L} refined; cand0 head: "
              f"{new_tokens[0][:10]}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
