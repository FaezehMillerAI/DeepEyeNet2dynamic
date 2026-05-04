from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .vocab import Vocabulary, build_concepts, build_vocab, normalize_concept


SPLIT_FILES = {
    "train": ("DeepEyeNet_train.json", "train.csv"),
    "valid": ("DeepEyeNet_valid.json", "valid.csv"),
    "test": ("DeepEyeNet_test.json", "test.csv"),
}


def _parse_keywords(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        loaded = json.loads(text.replace("'", '"'))
        if isinstance(loaded, list):
            return [str(v) for v in loaded]
    except Exception:
        pass
    return [part.strip() for part in re.split(r"[;,]", text) if part.strip()]


def _load_json_flexible(path: Path) -> Any:
    """Load strict JSON, JSONL, or concatenated JSON objects.

    Some Drive exports look like one JSON dictionary, while others contain one
    object per line or multiple JSON dictionaries written back-to-back. Python's
    plain json.loads raises "Extra data" for those concatenated exports.
    """
    text = path.read_text(encoding="utf-8-sig").strip()
    decoder = json.JSONDecoder()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    line_objects = []
    line_parse_failed = False
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            line_objects.append(json.loads(line))
        except json.JSONDecodeError:
            line_parse_failed = True
            break
    if line_objects and not line_parse_failed:
        return line_objects

    objects = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        objects.append(obj)
        idx = end
    if len(objects) == 1:
        return objects[0]
    return objects


def _append_record(records: list[dict[str, Any]], image_path: str, meta: dict[str, Any]) -> None:
    records.append(
        {
            "image_path": image_path,
            "keywords": _parse_keywords(meta.get("Keywords", meta.get("keywords", []))),
            "clinical_description": str(meta.get("clinical-description", meta.get("clinical_description", ""))),
            "report_text": str(meta.get("report_text", meta.get("clinical-description", meta.get("clinical_description", "")))),
        }
    )


def _records_from_json_raw(raw: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(raw, dict):
        for image_path, meta in raw.items():
            if isinstance(meta, dict):
                _append_record(records, str(image_path), meta)
        return records

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and "image_path" in item:
                _append_record(records, str(item["image_path"]), item)
            elif isinstance(item, dict):
                for image_path, meta in item.items():
                    if isinstance(meta, dict):
                        _append_record(records, str(image_path), meta)
        return records

    raise ValueError(f"Unsupported JSON metadata structure: {type(raw)!r}")


def load_split_records(data_root: str | Path, split: str) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    json_name, csv_name = SPLIT_FILES[split]
    json_path = data_root / json_name
    csv_path = data_root / csv_name

    records: list[dict[str, Any]] = []
    if json_path.exists():
        try:
            raw = _load_json_flexible(json_path)
            records = _records_from_json_raw(raw)
            if records:
                return records
        except Exception as exc:
            if not csv_path.exists():
                raise
            warnings.warn(f"Could not parse {json_path.name} ({exc}); falling back to {csv_path.name}.")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            records.append(
                {
                    "image_path": str(row["image_path"]),
                    "keywords": _parse_keywords(row.get("Keywords", [])),
                    "clinical_description": str(row.get("clinical-description", "")),
                    "report_text": str(row.get("report_text", row.get("clinical-description", ""))),
                }
            )
        return records

    raise FileNotFoundError(f"Could not find {json_path} or {csv_path}.")


def build_artifacts(
    data_root: str | Path,
    min_token_freq: int,
    max_vocab_size: int,
    max_concepts: int,
) -> tuple[Vocabulary, list[str]]:
    train = load_split_records(data_root, "train")
    vocab = build_vocab((r["report_text"] for r in train), min_token_freq, max_vocab_size)
    concepts = build_concepts((r["keywords"] for r in train), max_concepts=max_concepts)
    return vocab, concepts


def make_transforms(image_size: int, train: bool) -> Callable:
    aug = []
    if train:
        aug.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.08, contrast=0.08, saturation=0.05),
            ]
        )
    else:
        aug.extend([transforms.Resize((image_size, image_size))])
    aug.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(aug)


class DeepEyeNetDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        vocab: Vocabulary,
        concepts: list[str],
        image_size: int = 224,
        max_report_len: int = 96,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.records = load_split_records(data_root, split)
        self.vocab = vocab
        self.concepts = concepts
        self.concept_to_idx = {c: i for i, c in enumerate(concepts)}
        self.max_report_len = max_report_len
        self.transform = make_transforms(image_size, train=split == "train")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image_path = self.data_root / rec["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        token_ids = self.vocab.encode(rec["report_text"], self.max_report_len)
        concept_targets = torch.zeros(len(self.concepts), dtype=torch.float32)
        for kw in rec["keywords"]:
            concept = normalize_concept(kw)
            if concept in self.concept_to_idx:
                concept_targets[self.concept_to_idx[concept]] = 1.0

        return {
            "image": image,
            "tokens": torch.tensor(token_ids, dtype=torch.long),
            "concept_targets": concept_targets,
            "image_path": rec["image_path"],
            "report_text": rec["report_text"],
            "keywords": rec["keywords"],
        }


def collate_fn(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    concept_targets = torch.stack([item["concept_targets"] for item in batch])
    max_len = max(item["tokens"].numel() for item in batch)
    tokens = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    lengths = torch.tensor([item["tokens"].numel() for item in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        tokens[i, : item["tokens"].numel()] = item["tokens"]
    return {
        "image": images,
        "tokens": tokens,
        "lengths": lengths,
        "concept_targets": concept_targets,
        "image_path": [item["image_path"] for item in batch],
        "report_text": [item["report_text"] for item in batch],
        "keywords": [item["keywords"] for item in batch],
    }
