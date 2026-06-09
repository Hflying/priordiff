#!/usr/bin/env python3
"""Smoke test: gaussian_diffusion.training_losses_e2e + L_allusion 单步。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import BertConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from improved_diffusion.allusion_data import AllusionKnowledgeBase  # noqa: E402
from improved_diffusion.gaussian_diffusion import (  # noqa: E402
    ModelMeanType,
    ModelVarType,
    LossType,
    GaussianDiffusion,
    get_named_beta_schedule,
)
from improved_diffusion.transformer_model2 import TransformerNetModel2  # noqa: E402


class _SimpleWrap:
    def __init__(self, model):
        self.model = model

    def __call__(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)


def build_model(device):
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
    model.num_timesteps = 200
    model.allusion_loss_weight = 0.1
    return model


def build_diffusion():
    return GaussianDiffusion(
        betas=get_named_beta_schedule("sqrt", 200),
        model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL,
        loss_type=LossType.E2E_MSE,
        rescale_timesteps=True,
        model_arch="transformer",
        training_mode="e2e",
    )


def main():
    device = torch.device("cpu")
    model = build_model(device)
    model.train()

    kb = AllusionKnowledgeBase.from_json(ROOT.parent / "datasets/allusion_kb/allusion_v0.json")
    batch_sz = 2
    seq_len = 144
    allusion_kw = {}
    for key in (
        "allusion_text_ids", "allusion_text_mask",
        "allusion_semantic_ids", "allusion_semantic_mask",
        "allusion_tone_ids", "allusion_tone_mask",
    ):
        rows = [kb.tokenize_entry(kb.sample())[key] for _ in range(batch_sz)]
        allusion_kw[key] = torch.tensor(rows, dtype=torch.long, device=device)
    allusion_kw["allusion_sentiment"] = torch.zeros(batch_sz, device=device)
    allusion_kw["allusion_theme_id"] = torch.zeros(batch_sz, dtype=torch.long, device=device)

    x_start = torch.randn(batch_sz, seq_len, 16, device=device)
    t = torch.tensor([150, 30], device=device)
    model_kwargs = {
        "input_ids": torch.randint(4, 5000, (batch_sz, seq_len), device=device),
        **allusion_kw,
    }

    diffusion = build_diffusion()
    wrapped = _SimpleWrap(model)
    losses = diffusion.training_losses_e2e(wrapped, x_start, t, model_kwargs=model_kwargs)

    assert "loss" in losses
    assert "mse" in losses
    assert "loss_allusion" in losses, "L_allusion 未写入 terms"
    print(f"[ok] loss={losses['loss'].mean().item():.4f} "
          f"mse={losses['mse'].mean().item():.4f} "
          f"L_allusion={float(losses['loss_allusion']):.4f}")

    losses["loss"].mean().backward()
    assert model.allusion_poem_proj.weight.grad is not None
    print("[ok] backward through L_allusion")


if __name__ == "__main__":
    main()
