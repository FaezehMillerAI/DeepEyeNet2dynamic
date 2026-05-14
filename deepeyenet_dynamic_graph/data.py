from __future__ import annotations

import json
import re
import warnings
import random
from collections import Counter, defaultdict
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

IU_XRAY_TERMS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "effusion",
    "emphysema",
    "fibrosis",
    "fracture",
    "granuloma",
    "hernia",
    "hyperinflation",
    "infiltrate",
    "mass",
    "nodule",
    "opacity",
    "pneumonia",
    "pneumothorax",
    "scar",
    "tortuous aorta",
    "vascular congestion",
    "pleural thickening",
    "low lung volume",
    "calcified granuloma",
    "hiatal hernia",
]


ANATOMY_NODES = {
    "deepeyenet": ["superior retina", "inferior retina", "nasal retina", "temporal retina", "macula", "optic disc", "retinal vessels"],
    "iuxray": ["left upper lung", "left lower lung", "right upper lung", "right lower lung", "cardiac silhouette", "mediastinum", "pleura"],
}


def normalize_dataset_name(dataset: str) -> str:
    normalized = dataset.lower().replace("-", "").replace("_", "")
    if normalized == "deepeyenet":
        return "deepeyenet"
    if normalized in {"iuxray", "iuchestxray", "indianaxray"}:
        return "iuxray"
    raise ValueError(f"Unsupported dataset '{dataset}'. Use 'deepeyenet' or 'iuxray'.")


def get_anatomy_names(dataset: str) -> list[str]:
    return ANATOMY_NODES[normalize_dataset_name(dataset)]


def anatomy_prior_matrix(dataset: str, patch_grid: int) -> torch.Tensor:
    """Return a deterministic region-to-anatomy prior for weakly supervised data."""
    dataset = normalize_dataset_name(dataset)
    anatomy = get_anatomy_names(dataset)
    prior = torch.zeros(patch_grid * patch_grid, len(anatomy), dtype=torch.float32)
    for y in range(patch_grid):
        for x in range(patch_grid):
            r = y * patch_grid + x
            yn = (y + 0.5) / patch_grid
            xn = (x + 0.5) / patch_grid
            if dataset == "deepeyenet":
                names = {name: i for i, name in enumerate(anatomy)}
                prior[r, names["superior retina"]] = max(0.05, 1.0 - yn)
                prior[r, names["inferior retina"]] = max(0.05, yn)
                prior[r, names["nasal retina"]] = max(0.05, 1.0 - xn)
                prior[r, names["temporal retina"]] = max(0.05, xn)
                dist_macula = ((xn - 0.50) ** 2 + (yn - 0.50) ** 2) ** 0.5
                dist_disc = ((xn - 0.28) ** 2 + (yn - 0.50) ** 2) ** 0.5
                prior[r, names["macula"]] = max(0.05, 1.0 - 3.0 * dist_macula)
                prior[r, names["optic disc"]] = max(0.05, 1.0 - 3.0 * dist_disc)
                prior[r, names["retinal vessels"]] = max(0.05, 1.0 - 2.2 * abs(yn - 0.50))
            else:
                names = {name: i for i, name in enumerate(anatomy)}
                left = xn < 0.5
                upper = yn < 0.5
                lung_name = f"{'left' if left else 'right'} {'upper' if upper else 'lower'} lung"
                prior[r, names[lung_name]] = 1.0
                prior[r, names["cardiac silhouette"]] = max(0.05, 1.0 - 4.0 * (((xn - 0.50) ** 2 + (yn - 0.66) ** 2) ** 0.5))
                prior[r, names["mediastinum"]] = max(0.05, 1.0 - 5.0 * abs(xn - 0.50))
                border = min(xn, 1.0 - xn, yn, 1.0 - yn)
                prior[r, names["pleura"]] = max(0.05, 1.0 - 4.0 * border)
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return prior


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


def load_deepeyenet_split_records(data_root: str | Path, split: str) -> list[dict[str, Any]]:
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


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _iu_report_text(row: pd.Series) -> str:
    parts = []
    findings = _clean_text(row.get("findings", ""))
    impression = _clean_text(row.get("impression", ""))
    if findings:
        parts.append(f"Findings: {findings}")
    if impression:
        parts.append(f"Impression: {impression}")
    return " ".join(parts) or _clean_text(row.get("indication", ""))


def _iu_keywords(text: str, extra_terms: list[str] | None = None) -> list[str]:
    text_l = text.lower()
    terms = list(IU_XRAY_TERMS)
    if extra_terms:
        terms.extend(extra_terms)
    found = []
    for term in terms:
        norm = normalize_concept(term)
        if norm and re.search(rf"\b{re.escape(norm)}\b", text_l):
            found.append(norm)
    if any(phrase in text_l for phrase in ["no acute", "normal chest", "no active disease"]):
        found.append("no acute cardiopulmonary abnormality")
    return sorted(set(found))


def _find_iuxray_images_dir(data_root: Path) -> Path:
    candidates = [
        data_root / "images" / "images_normalized",
        data_root / "images_normalized",
        data_root / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find IU-XRay images directory under {data_root}.")


def _split_uids(uids: list[int], split: str, seed: int = 42) -> set[int]:
    rng = random.Random(seed)
    shuffled = list(uids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    train_end = int(0.70 * n)
    valid_end = train_end + int(0.15 * n)
    if split == "train":
        selected = shuffled[:train_end]
    elif split in {"valid", "val"}:
        selected = shuffled[train_end:valid_end]
    elif split == "test":
        selected = shuffled[valid_end:]
    else:
        raise ValueError(f"Unknown split: {split}")
    return set(selected)


def load_iuxray_split_records(data_root: str | Path, split: str, seed: int = 42) -> list[dict[str, Any]]:
    data_root = Path(data_root)
    reports_path = data_root / "indiana_reports.csv"
    if not reports_path.exists():
        raise FileNotFoundError(f"Could not find IU-XRay reports CSV at {reports_path}.")

    images_dir = _find_iuxray_images_dir(data_root)
    pattern = re.compile(r"(\d+)_IM-\d+-\d+\.dcm\.png")
    uid_to_images: dict[int, list[str]] = defaultdict(list)
    for image_file in images_dir.iterdir():
        match = pattern.match(image_file.name)
        if match:
            uid_to_images[int(match.group(1))].append(image_file.name)

    df = pd.read_csv(reports_path).copy()
    for col in ["findings", "impression", "indication", "comparison"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")
    df = df[(df["findings"].str.len() > 0) | (df["impression"].str.len() > 0)]
    df["image_files"] = df["uid"].apply(lambda uid: sorted(uid_to_images.get(int(uid), [])))
    df = df[df["image_files"].apply(len) > 0]
    selected_uids = _split_uids(sorted(df["uid"].astype(int).unique().tolist()), split, seed=seed)
    df = df[df["uid"].astype(int).isin(selected_uids)]

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        report = _iu_report_text(row)
        description = " ".join(
            part
            for part in [
                f"Indication: {_clean_text(row.get('indication', ''))}" if _clean_text(row.get("indication", "")) else "",
                report,
            ]
            if part
        )
        keywords = _iu_keywords(report)
        for image_name in row["image_files"]:
            image_path = images_dir.relative_to(data_root) / image_name
            records.append(
                {
                    "image_path": str(image_path),
                    "keywords": keywords,
                    "clinical_description": description,
                    "report_text": report,
                    "uid": int(row["uid"]),
                }
            )
    return records


def load_split_records(data_root: str | Path, split: str, dataset: str = "deepeyenet", seed: int = 42) -> list[dict[str, Any]]:
    dataset = normalize_dataset_name(dataset)
    if dataset == "deepeyenet":
        return load_deepeyenet_split_records(data_root, split)
    if dataset in {"iuxray", "iuchestxray", "indianaxray"}:
        return load_iuxray_split_records(data_root, split, seed=seed)
    raise ValueError(f"Unsupported dataset '{dataset}'. Use 'deepeyenet' or 'iuxray'.")


def apply_record_concepts(records: list[dict[str, Any]], per_record_concepts: dict[str, list[str]]) -> list[dict[str, Any]]:
    if not per_record_concepts:
        return records
    updated = []
    for rec in records:
        key = str(rec.get("uid", rec.get("image_path", "")))
        image_key = str(rec.get("image_path", ""))
        concepts = per_record_concepts.get(key) or per_record_concepts.get(image_key)
        if concepts:
            rec = dict(rec)
            rec["keywords"] = concepts
        updated.append(rec)
    return updated


def infer_concepts_from_reports(records: list[dict[str, Any]], max_concepts: int) -> list[str]:
    stopwords = {
        "there", "with", "without", "findings", "impression", "comparison", "indication",
        "normal", "image", "images", "view", "views", "year", "old", "patient", "female",
        "male", "right", "left", "chest", "lung", "lungs", "heart", "size", "clear",
        "mild", "moderate", "severe", "seen", "noted", "stable", "acute", "again",
        "exam", "portable", "frontal", "lateral", "study", "film", "evidence",
        "large", "small", "lower", "upper", "bilateral", "single", "present", "absent",
        "unchanged", "interval", "prior", "grossly", "within", "limits", "disease",
    }
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record["keywords"])
        text = str(record.get("report_text", "")).lower()
        for token in re.findall(r"[a-z][a-z-]{3,}", text):
            if token not in stopwords:
                counter[token] += 1
    return [term for term, _ in counter.most_common(max_concepts)]


def build_artifacts(
    data_root: str | Path,
    min_token_freq: int,
    max_vocab_size: int,
    max_concepts: int,
    dataset: str = "deepeyenet",
    seed: int = 42,
) -> tuple[Vocabulary, list[str]]:
    train = load_split_records(data_root, "train", dataset=dataset, seed=seed)
    vocab = build_vocab((r["report_text"] for r in train), min_token_freq, max_vocab_size)
    concepts = build_concepts((r["keywords"] for r in train), max_concepts=max_concepts)
    if not concepts:
        concepts = infer_concepts_from_reports(train, max_concepts=max_concepts)
    return vocab, concepts


def make_transforms(image_size: int, train: bool) -> Callable:
    aug = []
    if train:
        aug.extend(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
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


class MedicalReportDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        vocab: Vocabulary,
        concepts: list[str],
        dataset: str = "deepeyenet",
        image_size: int = 224,
        max_report_len: int = 96,
        seed: int = 42,
        per_record_concepts: dict[str, list[str]] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.dataset = dataset
        self.records = apply_record_concepts(load_split_records(data_root, split, dataset=dataset, seed=seed), per_record_concepts or {})
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


DeepEyeNetDataset = MedicalReportDataset


class HFMedicalReportDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str,
        tokenizer: Any,
        concepts: list[str],
        dataset: str = "deepeyenet",
        image_size: int = 224,
        max_report_len: int = 96,
        seed: int = 42,
        per_record_concepts: dict[str, list[str]] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.dataset = dataset
        self.records = apply_record_concepts(load_split_records(data_root, split, dataset=dataset, seed=seed), per_record_concepts or {})
        self.tokenizer = tokenizer
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
        encoded = self.tokenizer(
            rec["report_text"],
            truncation=True,
            max_length=self.max_report_len,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        if self.tokenizer.bos_token_id is not None and (not input_ids or input_ids[0] != self.tokenizer.bos_token_id):
            input_ids = [self.tokenizer.bos_token_id] + input_ids
            input_ids = input_ids[: self.max_report_len]
        if self.tokenizer.eos_token_id is not None and (not input_ids or input_ids[-1] != self.tokenizer.eos_token_id):
            input_ids = input_ids[: self.max_report_len - 1] + [self.tokenizer.eos_token_id]
        concept_targets = torch.zeros(len(self.concepts), dtype=torch.float32)
        for kw in rec["keywords"]:
            concept = normalize_concept(kw)
            if concept in self.concept_to_idx:
                concept_targets[self.concept_to_idx[concept]] = 1.0
        return {
            "image": image,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
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


def collate_hf_fn(batch: list[dict[str, Any]], pad_id: int) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch])
    concept_targets = torch.stack([item["concept_targets"] for item in batch])
    max_len = max(item["input_ids"].numel() for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].numel()
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = 1
    return {
        "image": images,
        "tokens": input_ids,
        "attention_mask": attention_mask,
        "lengths": attention_mask.sum(dim=1),
        "concept_targets": concept_targets,
        "image_path": [item["image_path"] for item in batch],
        "report_text": [item["report_text"] for item in batch],
        "keywords": [item["keywords"] for item in batch],
    }
