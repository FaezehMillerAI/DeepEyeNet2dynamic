from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .utils import load_json, save_json
from .vocab import normalize_concept


def _record_concepts(record: dict[str, Any]) -> list[str]:
    return sorted({normalize_concept(c) for c in record.get("keywords", []) if normalize_concept(c)})


def build_report_memory(records: list[dict[str, Any]], max_entries: int = 2500) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    meta: dict[str, dict[str, Any]] = {}
    for record in records:
        report = " ".join(str(record.get("report_text", "")).split())
        if not report:
            continue
        counter[report] += 1
        concepts = set(meta.get(report, {}).get("concepts", []))
        concepts.update(_record_concepts(record))
        meta[report] = {"report": report, "concepts": sorted(concepts)}
    entries = []
    for report, count in counter.most_common(max_entries):
        entry = dict(meta[report])
        entry["count"] = count
        entries.append(entry)
    return entries


def save_report_memory(records: list[dict[str, Any]], output_path: str | Path, max_entries: int = 2500) -> list[dict[str, Any]]:
    memory = build_report_memory(records, max_entries=max_entries)
    save_json({"entries": memory}, output_path)
    return memory


def load_report_memory(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    raw = load_json(path)
    if isinstance(raw, dict):
        return [e for e in raw.get("entries", []) if isinstance(e, dict)]
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    return []


def concept_set_from_probs(concepts: list[str], probs, threshold: float = 0.35, top_k: int = 5) -> set[str]:
    values = probs.tolist() if hasattr(probs, "tolist") else list(probs)
    selected = {concepts[i] for i, p in enumerate(values) if p >= threshold and i < len(concepts)}
    if not selected and values:
        top_ids = sorted(range(min(len(values), len(concepts))), key=lambda i: values[i], reverse=True)[:top_k]
        selected.update(concepts[i] for i in top_ids)
    return {normalize_concept(c) for c in selected if normalize_concept(c)}


_NORMAL_CONCEPTS = {
    "normal",
    "no acute abnormality",
    "no acute cardiopulmonary abnormality",
    "no acute cardiopulmonary disease",
    "no acute disease",
    "no active disease",
    "clear lungs",
    "lungs clear",
}


def _is_normal_or_absent_concept(concept: str) -> bool:
    concept = normalize_concept(concept)
    return (
        not concept
        or concept in _NORMAL_CONCEPTS
        or concept.startswith("no ")
        or concept.startswith("without ")
        or concept.startswith("negative for ")
        or concept.startswith("absence of ")
    )


def abnormal_concepts(concepts: set[str]) -> set[str]:
    return {normalize_concept(c) for c in concepts if not _is_normal_or_absent_concept(c)}


def retrieve_report(
    memory: list[dict[str, Any]],
    query_concepts: set[str],
    min_score: float = 0.05,
) -> tuple[str | None, float]:
    if not memory or not query_concepts:
        return None, 0.0
    query_abnormal = abnormal_concepts(query_concepts)
    best_report = None
    best_score = 0.0
    for entry in memory:
        entry_concepts = {normalize_concept(c) for c in entry.get("concepts", []) if normalize_concept(c)}
        if not entry_concepts:
            continue
        entry_abnormal = abnormal_concepts(entry_concepts)
        if query_abnormal and not (query_abnormal & entry_abnormal):
            continue
        overlap = len(query_concepts & entry_concepts)
        union = len(query_concepts | entry_concepts)
        jaccard = overlap / max(1, union)
        abnormal_overlap = len(query_abnormal & entry_abnormal)
        abnormal_bonus = 0.15 * abnormal_overlap
        frequency_bonus = min(0.10, 0.01 * float(entry.get("count", 1)))
        score = jaccard + abnormal_bonus + frequency_bonus
        if score > best_score:
            best_score = score
            best_report = str(entry.get("report", ""))
    if best_report and best_score >= min_score:
        return best_report, best_score
    return None, best_score
