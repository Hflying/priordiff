"""典故对齐对比损失 L_allusion。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .allusion_encoder import pool_hidden_states


class AllusionContrastiveLoss(nn.Module):
    """InfoNCE：诗句表征应接近目标典故、远离 batch 内负样本。

    poem_repr: 扩散模型 decoder 输出的 masked mean pool [B, D]
    allusion_hidden: AllusionEncoder 输出 [B, S, D]
    allusion_mask: [B, S]
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        poem_repr: torch.Tensor,
        allusion_hidden: torch.Tensor,
        allusion_mask: torch.Tensor,
    ) -> torch.Tensor:
        allusion_repr = pool_hidden_states(allusion_hidden, allusion_mask)
        if poem_repr.shape[-1] != allusion_repr.shape[-1]:
            poem_repr = F.layer_norm(poem_repr, (poem_repr.shape[-1],))
            allusion_repr = F.layer_norm(allusion_repr, (allusion_repr.shape[-1],))

        poem_n = F.normalize(poem_repr, dim=-1)
        all_n = F.normalize(allusion_repr, dim=-1)
        logits = poem_n @ all_n.t() / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)
        return 0.5 * (loss_i + loss_t)


class AllusionLossBundle(nn.Module):
    """L = α·L_sem + β·L_metric + γ·L_allusion 的典故部分封装。"""

    def __init__(self, temperature: float = 0.07, weight: float = 1.0):
        super().__init__()
        self.contrastive = AllusionContrastiveLoss(temperature=temperature)
        self.weight = weight

    def forward(self, poem_repr, allusion_hidden, allusion_mask) -> dict:
        loss = self.contrastive(poem_repr, allusion_hidden, allusion_mask) * self.weight
        return {"loss_allusion": loss}
