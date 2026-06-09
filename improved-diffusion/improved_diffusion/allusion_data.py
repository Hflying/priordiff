"""典故知识库加载与 batch 字段构造。"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from improved_diffusion.allusion_encoder import TONE2ID, encode_tone_pattern

PAD_ID = 0
UNK_ID = 1


@dataclass
class AllusionEntry:
    id: str
    surface_form: str
    semantic: str
    sentiment: float
    theme: list[str]
    tone_pattern: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AllusionEntry":
        return cls(
            id=d["id"],
            surface_form=d.get("surface_form", ""),
            semantic=d.get("semantic", ""),
            sentiment=float(d.get("sentiment", 0.0)),
            theme=list(d.get("theme") or []),
            tone_pattern=d.get("tone_pattern", ""),
        )


class CharVocab:
    def __init__(self):
        self.char2id: dict[str, int] = {"<PAD>": PAD_ID, "<UNK>": UNK_ID}

    def add_text(self, text: str) -> None:
        for ch in text:
            if ch not in self.char2id:
                self.char2id[ch] = len(self.char2id)

    def encode(self, text: str, max_len: int) -> tuple[list[int], list[int]]:
        ids, mask = [], []
        for ch in text[:max_len]:
            ids.append(self.char2id.get(ch, UNK_ID))
            mask.append(1)
        while len(ids) < max_len:
            ids.append(PAD_ID)
            mask.append(0)
        return ids, mask

    @property
    def size(self) -> int:
        return len(self.char2id)


class AllusionKnowledgeBase:
    def __init__(self, entries: list[AllusionEntry], char_vocab: CharVocab | None = None):
        self.entries = entries
        self.char_vocab = char_vocab or CharVocab()
        for e in entries:
            self.char_vocab.add_text(e.surface_form)
            self.char_vocab.add_text(e.semantic)

    @classmethod
    def from_json(cls, path: str | Path) -> "AllusionKnowledgeBase":
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw.get("entries", raw if isinstance(raw, list) else [])
        entries = [AllusionEntry.from_dict(x) for x in items if x.get("surface_form")]
        return cls(entries)

    def sample(self, rng: random.Random | None = None) -> AllusionEntry:
        rng = rng or random.Random()
        return rng.choice(self.entries)

    def theme_id(self, entry: AllusionEntry, num_themes: int = 128) -> int:
        key = entry.theme[0] if entry.theme else entry.id
        h = int(hashlib.md5(key.encode()).hexdigest(), 16)
        return h % num_themes

    def tokenize_entry(
        self,
        entry: AllusionEntry,
        max_text_len: int = 16,
        max_semantic_len: int = 24,
        max_tone_len: int = 8,
        num_themes: int = 128,
    ) -> dict[str, list | float | int]:
        text_ids, text_mask = self.char_vocab.encode(entry.surface_form, max_text_len)
        sem_ids, sem_mask = self.char_vocab.encode(entry.semantic, max_semantic_len)
        tone_ids, tone_mask = encode_tone_pattern(entry.tone_pattern, max_tone_len)
        return {
            "allusion_text_ids": text_ids,
            "allusion_text_mask": text_mask,
            "allusion_semantic_ids": sem_ids,
            "allusion_semantic_mask": sem_mask,
            "allusion_sentiment": entry.sentiment,
            "allusion_tone_ids": tone_ids,
            "allusion_tone_mask": tone_mask,
            "allusion_theme_id": self.theme_id(entry, num_themes),
            "allusion_entry_id": entry.id,
        }


def attach_allusion_fields(
    train_lst: list[dict],
    kb_path: str | Path,
    seed: int = 42,
) -> list[dict]:
    """为每条训练样本随机绑定一条典故条件（弱监督占位）。"""
    kb = AllusionKnowledgeBase.from_json(kb_path)
    rng = random.Random(seed)
    out = []
    for rec in train_lst:
        entry = kb.sample(rng)
        fields = kb.tokenize_entry(entry)
        merged = dict(rec)
        merged.update(fields)
        out.append(merged)
    return out


def resolve_allusion_entry(kb: AllusionKnowledgeBase, entry_id: str | None) -> AllusionEntry:
    if entry_id:
        for entry in kb.entries:
            if entry.id == entry_id:
                return entry
    return kb.sample()


def build_allusion_model_kwargs(
    kb_path: str | Path,
    batch_size: int,
    device: torch.device | str,
    entry_id: str | None = None,
    allusion_pii: bool = True,
) -> dict[str, torch.Tensor | bool]:
    """将单条典故条件广播为 infill / 采样用的 model_kwargs。"""
    kb = AllusionKnowledgeBase.from_json(kb_path)
    entry = resolve_allusion_entry(kb, entry_id)
    fields = kb.tokenize_entry(entry)
    device = torch.device(device)
    out: dict[str, torch.Tensor | bool] = {'allusion_pii': allusion_pii}
    for key, val in fields.items():
        if key == 'allusion_entry_id':
            continue
        if key == 'allusion_sentiment':
            out[key] = torch.full((batch_size,), float(val), dtype=torch.float32, device=device)
        elif key == 'allusion_theme_id':
            out[key] = torch.full((batch_size,), int(val), dtype=torch.long, device=device)
        else:
            out[key] = torch.tensor([val], dtype=torch.long, device=device).expand(batch_size, -1)
    return out
