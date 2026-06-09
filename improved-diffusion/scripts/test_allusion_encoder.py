#!/usr/bin/env python3
"""Smoke test: AllusionEncoder + TransformerNetModel2 allusion_gen forward."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import BertConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from improved_diffusion.allusion_data import AllusionKnowledgeBase  # noqa: E402
from improved_diffusion.allusion_encoder import progressive_injection_gamma  # noqa: E402
from improved_diffusion.transformer_model2 import TransformerNetModel2  # noqa: E402


def main():
    device = torch.device("cpu")
    config = BertConfig(
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=512,
        hidden_dropout_prob=0.1,
    )

    model = TransformerNetModel2(
        in_channels=16,
        model_channels=128,
        out_channels=16,
        num_res_blocks=2,
        attention_resolutions=(4,),
        dropout=0.1,
        config=config,
        training_mode="e2e",
        vocab_size=5049,
        experiment_mode="allusion_gen",
        logits_mode=1,
    ).to(device)
    model.num_timesteps = 2000
    model.eval()

    kb = AllusionKnowledgeBase.from_json(ROOT.parent / "datasets/allusion_kb/allusion_v0.json")
    batch = []
    for _ in range(2):
        batch.append(kb.tokenize_entry(kb.sample()))

    def collate(key, dtype=torch.long):
        if key == "allusion_sentiment":
            return torch.tensor([b[key] for b in batch], dtype=torch.float32, device=device)
        if key == "allusion_theme_id":
            return torch.tensor([b[key] for b in batch], dtype=torch.long, device=device)
        return torch.tensor([b[key] for b in batch], dtype=dtype, device=device)

    kwargs = {
        "allusion_text_ids": collate("allusion_text_ids"),
        "allusion_text_mask": collate("allusion_text_mask"),
        "allusion_semantic_ids": collate("allusion_semantic_ids"),
        "allusion_semantic_mask": collate("allusion_semantic_mask"),
        "allusion_sentiment": collate("allusion_sentiment"),
        "allusion_tone_ids": collate("allusion_tone_ids"),
        "allusion_tone_mask": collate("allusion_tone_mask"),
        "allusion_theme_id": collate("allusion_theme_id"),
    }

    x = torch.randn(2, 144, 16, device=device)
    t = torch.tensor([1500, 200], device=device)
    gamma = progressive_injection_gamma(t, 2000)
    print("PII gamma:", gamma.tolist())

    with torch.no_grad():
        out = model(x, t, **kwargs)
    print("output shape:", tuple(out.shape))
    assert out.shape == x.shape
    print("[ok] allusion_gen forward pass")

    model.train()
    pred_xstart = torch.randn(2, 144, 16, device=device, requires_grad=True)
    loss = model.compute_allusion_training_loss(pred_xstart, t, kwargs)
    assert loss is not None and loss.ndim == 0
    loss.backward()
    assert pred_xstart.grad is not None
    print(f"[ok] L_allusion={loss.item():.4f}")


if __name__ == "__main__":
    main()
