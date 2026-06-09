"""Sonnet-side M3-v1 refinement (parallel to ``decode_pkl_m3_v1.py``).

Inputs:
  --lcvr_json   the JSON written by ``sonnet_sample_lcvr.py``
                (keys: ``baseline``, ``lcvr[idx_str]["tokens"]``).
  --target_json target_son10.json (one record per line).
  --mlm_ckpt    path to ``mlm_final.pt`` produced by
                ``train_m3v1_mlm_sonnet.py``.
  --vocab_dir   sonnet diffusion checkpoint dir (must contain vocab.json
                used both for the diffusion logits and for the MLM).
  --out         path to write augmented JSON (schema = input + new
                ``m3v1[idx_str]["tokens"]`` field).

The procedure follows the ci version: identify word-slot positions
(``i ≤ last_triplet_end and i not in triplet``), iteratively detect
tokens whose in-cand count ≥ ``--repeat_threshold``, mask them, run a
single MLM forward pass, and ``argmax`` under the position-specific
LCVR mask.  Triplet anchors and tail PADs are never modified.

Usage:
  cd improved-diffusion
  TRANSFORMERS_OFFLINE=1 python scripts/decode_sonnet_m3_v1.py \\
      --lcvr_json out_gen/sonnet_son10_25000.json \\
      --target_json control_gen/target_son10.json \\
      --mlm_ckpt diffusion_models/m3v1_mlm_sonnet3355/mlm_final.pt \\
      --vocab_dir diffusion_models/diff_e2e-tgt_pad_2000steps_16dim_sonnet3355 \\
      --out out_gen/sonnet_son10_25000_m3v1.json
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import torch as th

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFORMERS_SRC = REPO_ROOT / "transformers" / "src"
if TRANSFORMERS_SRC.is_dir():
    sys.path.insert(0, str(TRANSFORMERS_SRC))

from transformers import BertConfig, BertForMaskedLM  # noqa: E402

SPECIAL_BASE = {"START", "END", "UNK", "PAD"}
LINE_TRIPLET = {"<", "eos", ">"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lcvr_json", required=True)
    p.add_argument("--target_json", required=True)
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


def build_mlm_position_mask(record, T: int, vocab: dict, V_mlm: int):
    """Same semantics as sonnet_sample_lcvr.build_position_masks but
    sized to the MLM vocabulary (which is smaller — typically
    2869 vs the diffusion lm_head's 22970)."""
    V_real = len(vocab)  # 2868
    mask = th.zeros(T, V_mlm, dtype=th.bool)
    pad_id = vocab["PAD"]
    lt_id = vocab.get("<")
    eos_word_id = vocab.get("eos")
    gt_id = vocab.get(">")
    forbidden_in_word = {
        vocab[t] for t in (SPECIAL_BASE | LINE_TRIPLET) if t in vocab
    }
    word_mask_row = th.zeros(V_mlm, dtype=th.bool)
    upper = min(V_real, V_mlm)
    word_mask_row[:upper] = True
    for tid in forbidden_in_word:
        if tid < V_mlm:
            word_mask_row[tid] = False

    last_triplet_end = max(record["gt_positions"])
    fixed = {}
    for p in record["lt_positions"]:
        fixed[p] = lt_id
    for p in record["eos_positions"]:
        fixed[p] = eos_word_id
    for p in record["gt_positions"]:
        fixed[p] = gt_id

    for i in range(T):
        if i in fixed:
            tid = fixed[i]
            if tid is not None and tid < V_mlm:
                mask[i, tid] = True
        elif i > last_triplet_end:
            if pad_id < V_mlm:
                mask[i, pad_id] = True
        else:
            mask[i] = word_mask_row
    return mask


def tokens_to_ids(token_list, vocab, unk_id, pad_id, T):
    """Convert one candidate (list[str] of length T) to long tensor (T,)."""
    ids = []
    for tok in token_list[:T]:
        ids.append(int(vocab.get(tok, unk_id)))
    while len(ids) < T:
        ids.append(pad_id)
    return th.tensor(ids, dtype=th.long)


def refine_batch(
    ids: th.Tensor,
    record,
    pmask: th.Tensor,
    mlm: BertForMaskedLM,
    mask_id: int,
    pad_id: int,
    repeat_threshold: int,
    max_iters: int,
    device: th.device,
):
    """ids: (B, T). 多候选共享同一 target 的 word_slots/pmask.

    Vectorised: repetition detection via per-row scatter_add_ over the
    MLM vocabulary, MLM forward + masked argmax done as a single
    (B, |W|, V_mlm) tensor op.
    """
    B, T = ids.shape
    last_triplet_end = max(record["gt_positions"])
    triplet_set = (
        set(record["lt_positions"])
        | set(record["eos_positions"])
        | set(record["gt_positions"])
    )
    word_positions = [i for i in range(last_triplet_end + 1) if i not in triplet_set]
    if not word_positions:
        return ids
    Vlog = mlm.config.vocab_size
    pmask = pmask.to(device)
    if pmask.size(1) != Vlog:
        raise ValueError(
            f"pmask vocab dim {pmask.size(1)} != MLM vocab_size {Vlog}"
        )

    ids = ids.clone().to(device)
    word_idx = th.tensor(word_positions, device=device, dtype=th.long)  # (|W|,)
    pm_word = pmask.index_select(0, word_idx).unsqueeze(0)  # (1, |W|, V_mlm)

    for it in range(max_iters):
        ids_word = ids.index_select(1, word_idx)  # (B, |W|)
        cnt_batch = th.zeros(B, Vlog, device=device, dtype=th.float)
        cnt_batch.scatter_add_(
            1, ids_word, th.ones_like(ids_word, dtype=th.float)
        )
        cnt_per_pos = cnt_batch.gather(1, ids_word)  # (B, |W|)
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
            logits = mlm(input_ids=inp, attention_mask=attn).logits  # (B, T, V)
        logits_word = logits.index_select(1, word_idx)  # (B, |W|, V)
        logits_word = logits_word.masked_fill(~pm_word, float("-inf"))
        new_ids_word = logits_word.argmax(dim=-1)  # (B, |W|)
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
    with open(args.target_json, "r", encoding="utf-8") as f:
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
        cands_tokens = data["lcvr"][idx]["tokens"]  # list[list[str]] (B, T)
        T = len(cands_tokens[0])
        rows = [tokens_to_ids(c, vocab, unk_id, pad_id, T) for c in cands_tokens]
        ids = th.stack(rows, dim=0)
        pmask = build_mlm_position_mask(tgt, T, vocab, V_mlm)
        refined = refine_batch(
            ids, tgt, pmask, mlm, mask_id, pad_id,
            args.repeat_threshold, args.refine_iters, device,
        )
        new_tokens = []
        for row in refined.tolist():
            new_tokens.append([inv_vocab.get(t, f"<{t}>") for t in row])
        out_data["m3v1"][idx] = {
            "src_idx": tgt["src_idx"],
            "n_tokens": tgt["n_tokens"],
            "tokens": new_tokens,
        }
        print(f"  [{idx}] refined; cand0 head sample: "
              f"{new_tokens[0][:8]}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"[done] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
