from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .vocab import normalize_concept


NORMALIZATION_RULES = {
    "amd": "age related macular degeneration",
    "dry amd": "dry age related macular degeneration",
    "wet amd": "neovascular age related macular degeneration",
    "pic": "punctate inner choroidopathy",
    "mewds": "multiple evanescent white dot syndrome",
    "pcv": "polypoidal choroidal vasculopathy",
    "pdr": "proliferative diabetic retinopathy",
    "nvd": "neovascularization of the disc",
    "srnv": "subretinal neovascularization",
    "srnv-md": "subretinal neovascular membrane",
    "pe folds": "pigment epithelial folds",
    "cardiac enlargement": "cardiomegaly",
    "enlarged cardiac silhouette": "cardiomegaly",
    "heart enlarged": "cardiomegaly",
    "airspace disease": "airspace opacity",
    "infiltrate": "pulmonary infiltrate",
    "pleural fluid": "pleural effusion",
    "no acute disease": "no acute cardiopulmonary abnormality",
    "normal chest": "no acute cardiopulmonary abnormality",
}

CLINICAL_FALLBACK_TERMS = [
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
    "vascular congestion",
    "pleural thickening",
    "macular hole",
    "cone dystrophy",
    "morning glory syndrome",
    "cavernous hemangioma of the retina",
    "uveitis",
    "tumor",
    "suprachoroidal hemorrhage",
]


def rule_normalize_concept(text: str) -> str:
    concept = normalize_concept(str(text))
    concept = re.sub(r"\b(left|right|bilateral|mild|moderate|severe|small|large)\b", "", concept)
    concept = re.sub(r"\s+", " ", concept).strip()
    return NORMALIZATION_RULES.get(concept, concept)


def llm_normalize_concepts(concepts: list[str], cache_path: str | Path, model: str = "gpt-4o-mini") -> dict[str, str]:
    """Optionally normalize concepts with an OpenAI-compatible LLM and cache results.

    This is intentionally explicit and cache-backed: training should not make
    hidden network calls unless the user passes the relevant CLI option and has
    configured an API key.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    else:
        cache = {}
    missing = [c for c in concepts if c not in cache]
    if not missing:
        return {c: cache[c] for c in concepts}
    if not os.environ.get("OPENAI_API_KEY"):
        for c in missing:
            cache[c] = rule_normalize_concept(c)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2))
        return {c: cache[c] for c in concepts}

    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            "Normalize these medical concepts to concise canonical names. "
            "Preserve clinically important disease/finding meaning. Return only JSON mapping original strings to canonical strings.\n"
            + json.dumps(missing, ensure_ascii=False)
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content or "{}"
        parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
        for c in missing:
            cache[c] = rule_normalize_concept(parsed.get(c, c))
    except Exception:
        for c in missing:
            cache[c] = rule_normalize_concept(c)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    return {c: cache[c] for c in concepts}


def _record_key(record: dict[str, Any]) -> str:
    if "uid" in record:
        return str(record["uid"])
    return str(record.get("image_path", ""))


def _iter_radgraph_docs(radgraph_path: str | Path) -> list[tuple[str, dict[str, Any]]]:
    path = Path(radgraph_path)
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        return [(str(k), v) for k, v in raw.items() if isinstance(v, dict)]
    if isinstance(raw, list):
        docs = []
        for i, item in enumerate(raw):
            if isinstance(item, dict):
                key = item.get("doc_key") or item.get("id") or item.get("uid") or item.get("image_path") or str(i)
                docs.append((str(key), item))
        return docs
    return []


def _extract_doc_entities(doc: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    entities_raw = doc.get("entities", {})
    if isinstance(entities_raw, list):
        items = [(str(i), e) for i, e in enumerate(entities_raw) if isinstance(e, dict)]
    else:
        items = [(str(k), v) for k, v in entities_raw.items() if isinstance(v, dict)]
    entities = []
    relations = []
    for eid, ent in items:
        tokens = ent.get("tokens") or ent.get("text") or ent.get("span") or ent.get("mention")
        if isinstance(tokens, list):
            text = " ".join(str(t) for t in tokens)
        else:
            text = str(tokens or "")
        label = str(ent.get("label") or ent.get("type") or "")
        canonical = rule_normalize_concept(text)
        if canonical:
            entities.append({"id": eid, "text": text, "canonical": canonical, "label": label})
        for rel in ent.get("relations", []) or []:
            if isinstance(rel, list) and len(rel) >= 2:
                relations.append({"source": eid, "target": str(rel[-1]), "type": str(rel[0])})
            elif isinstance(rel, dict):
                relations.append({"source": eid, "target": str(rel.get("target", "")), "type": str(rel.get("type", ""))})
    return entities, relations


def build_concept_graph(
    records: list[dict[str, Any]],
    max_concepts: int,
    radgraph_path: str | Path | None = None,
    normalizer: str = "rules",
    normalizer_cache: str | Path | None = None,
    llm_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    relation_counter: Counter[tuple[str, str, str]] = Counter()
    per_record: dict[str, list[str]] = {}

    rad_docs = dict(_iter_radgraph_docs(radgraph_path)) if radgraph_path else {}
    for record in records:
        key = _record_key(record)
        raw_terms = list(record.get("keywords", []))
        doc = rad_docs.get(key) or rad_docs.get(str(record.get("image_path", "")))
        id_to_concept = {}
        if doc:
            entities, relations = _extract_doc_entities(doc)
            raw_terms.extend(e["canonical"] for e in entities)
            id_to_concept = {e["id"]: e["canonical"] for e in entities}
            for rel in relations:
                src = id_to_concept.get(rel["source"])
                tgt = id_to_concept.get(rel["target"])
                if src and tgt:
                    relation_counter[(src, rel["type"] or "related_to", tgt)] += 1
        if not raw_terms:
            text = str(record.get("report_text", "")).lower()
            raw_terms.extend(term for term in CLINICAL_FALLBACK_TERMS if re.search(rf"\b{re.escape(term)}\b", text))
        normalized = [rule_normalize_concept(t) for t in raw_terms]
        normalized = [t for t in normalized if t]
        counter.update(normalized)
        per_record[key] = sorted(set(normalized))

    concepts = [c for c, _ in counter.most_common(max_concepts)]
    if normalizer == "llm" and concepts:
        cache = normalizer_cache or "outputs/concept_normalization_cache.json"
        mapping = llm_normalize_concepts(concepts, cache, model=llm_model)
        remapped_counter: Counter[str] = Counter()
        for concept, count in counter.items():
            remapped_counter[mapping.get(concept, concept)] += count
        concepts = [c for c, _ in remapped_counter.most_common(max_concepts)]
        remapped_relations = Counter()
        for (src, rel, tgt), count in relation_counter.items():
            remapped_relations[(mapping.get(src, src), rel, mapping.get(tgt, tgt))] += count
        relation_counter = remapped_relations
        for key, vals in per_record.items():
            per_record[key] = sorted(set(mapping.get(v, v) for v in vals))

    concept_set = set(concepts)
    relations_out = [
        {"source": src, "type": rel, "target": tgt, "count": count}
        for (src, rel, tgt), count in relation_counter.most_common()
        if src in concept_set and tgt in concept_set
    ]
    return {
        "concepts": concepts,
        "relations": relations_out,
        "per_record_concepts": per_record,
        "source": "radgraph" if rad_docs else "keywords_or_lexicon",
        "normalizer": normalizer,
    }
