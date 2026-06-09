"""Allusion encoder: 多维典故条件编码，输出 cross-attention 用的 hidden states."""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

TONE2ID = {"PING": 0, "ZE": 1, "NONE": 2, "PAD": 3, "平": 0, "仄": 1}


def progressive_injection_gamma(
    timesteps: torch.Tensor,
    num_timesteps: int,
    start_frac: float = 0.7,
    mid_frac: float = 0.3,
) -> torch.Tensor:
    """PII 权重 γ(t)：高噪声段不注入，低噪声段全注入。

    timesteps ∈ [0, T-1]，数值越大噪声越多。
    - t/T ≥ start_frac  → γ = 0
    - t/T ≤ mid_frac    → γ = 1
    - 中间线性插值
    """
    denom = max(num_timesteps - 1, 1)
    ratio = timesteps.float() / denom
    ramp = (start_frac - ratio) / max(start_frac - mid_frac, 1e-6)
    gamma = torch.where(
        ratio <= mid_frac,
        torch.ones_like(ratio),
        torch.where(ratio >= start_frac, torch.zeros_like(ratio), ramp.clamp(0.0, 1.0)),
    )
    return gamma


def encode_tone_pattern(pattern: str, max_len: int) -> tuple[list[int], list[int]]:
    """将 '平平仄仄' 或 'PING ZE ...' 转为 id 序列。"""
    ids, mask = [], []
    if not pattern:
        return [TONE2ID["PAD"]] * max_len, [0] * max_len
    tokens = []
    for ch in pattern.replace(" ", ""):
        if ch in ("平", "P"):
            tokens.append("PING")
        elif ch in ("仄", "Z"):
            tokens.append("ZE")
        elif ch.upper() in TONE2ID:
            tokens.append(ch.upper())
    for tok in tokens[:max_len]:
        ids.append(TONE2ID.get(tok, TONE2ID["NONE"]))
        mask.append(1)
    while len(ids) < max_len:
        ids.append(TONE2ID["PAD"])
        mask.append(0)
    return ids, mask


class CharTextEncoder(nn.Module):
    """字符级文本编码：Embedding + 单层 TransformerEncoderLayer 风格 MLP。"""

    def __init__(self, vocab_size: int, hidden_size: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        pos = torch.arange(seqlen, device=input_ids.device).unsqueeze(0).expand(bsz, -1)
        x = self.embed(input_ids) + self.pos(pos)
        x = self.dropout(self.norm(x))
        x = x + self.dropout(self.ff(x))
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        pooled = (x * mask).sum(dim=1) / denom
        return pooled


class AllusionEncoder(nn.Module):
    """融合四类典故特征 → [B, num_slots, hidden_size] 供 cross-attention 使用。

    Slots: text | semantic | sentiment | tone
    """

    NUM_SLOTS = 4

    def __init__(
        self,
        hidden_size: int = 768,
        char_vocab_size: int = 8192,
        max_text_len: int = 16,
        max_semantic_len: int = 24,
        max_tone_len: int = 8,
        num_themes: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_text_len = max_text_len
        self.max_semantic_len = max_semantic_len
        self.max_tone_len = max_tone_len

        self.text_enc = CharTextEncoder(char_vocab_size, hidden_size, max_text_len, dropout)
        self.semantic_enc = CharTextEncoder(char_vocab_size, hidden_size, max_semantic_len, dropout)
        self.sentiment_proj = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, hidden_size),
        )
        self.theme_emb = nn.Embedding(num_themes, hidden_size)
        self.tone_emb = nn.Embedding(len(TONE2ID), hidden_size, padding_idx=TONE2ID["PAD"])
        self.tone_pool = nn.Linear(hidden_size, hidden_size)

        self.slot_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(self.NUM_SLOTS)])
        self.slot_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def encode_tone_sequence(
        self, tone_ids: torch.Tensor, tone_mask: torch.Tensor
    ) -> torch.Tensor:
        emb = self.tone_emb(tone_ids)
        mask = tone_mask.unsqueeze(-1).float()
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.tone_pool(pooled)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        semantic_ids: torch.Tensor,
        semantic_mask: torch.Tensor,
        sentiment: torch.Tensor,
        tone_ids: torch.Tensor,
        tone_mask: torch.Tensor,
        theme_id: torch.Tensor,
        gamma: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            hidden_states: [B, 4, hidden_size]
            attention_mask: [B, 4] (1=valid)
        """
        text_vec = self.text_enc(text_ids, text_mask)
        sem_vec = self.semantic_enc(semantic_ids, semantic_mask)
        sent_vec = self.sentiment_proj(sentiment.unsqueeze(-1))
        tone_vec = self.encode_tone_sequence(tone_ids, tone_mask)
        theme_vec = self.theme_emb(theme_id.clamp(min=0))

        # theme 与 sentiment 融合到各自 slot；tone slot 叠加 theme 偏置
        slots = [
            self.slot_proj[0](text_vec),
            self.slot_proj[1](sem_vec + 0.5 * theme_vec),
            self.slot_proj[2](sent_vec + 0.5 * theme_vec),
            self.slot_proj[3](tone_vec),
        ]
        hidden = torch.stack(slots, dim=1)
        hidden = self.dropout(self.slot_norm(hidden))

        if gamma is not None:
            hidden = hidden * gamma.view(-1, 1, 1)

        attn_mask = torch.ones(
            hidden.shape[0], self.NUM_SLOTS, device=hidden.device, dtype=text_mask.dtype
        )
        return hidden, attn_mask

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True


def pool_hidden_states(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """对 encoder hidden 做 masked mean pool → [B, D]。"""
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)
