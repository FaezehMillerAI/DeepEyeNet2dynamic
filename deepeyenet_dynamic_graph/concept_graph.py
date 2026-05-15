from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .data import get_anatomy_names, normalize_dataset_name
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

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bno evidence of\b",
    r"\bno focal\b",
    r"\babsent\b",
]

NORMAL_PATTERNS = [
    r"\bnormal\b",
    r"\bunremarkable\b",
    r"\bwithin normal limits\b",
    r"\bclear\b",
]

ANATOMY_ALIASES = {
    "iuxray": {
        "left upper lung": ["left upper lung", "left upper lobe", "left apex"],
        "left lower lung": ["left lower lung", "left lower lobe", "left base", "left basilar"],
        "right upper lung": ["right upper lung", "right upper lobe", "right apex"],
        "right lower lung": ["right lower lung", "right lower lobe", "right base", "right basilar"],
        "cardiac silhouette": ["heart", "cardiac silhouette", "cardiomediastinal silhouette"],
        "mediastinum": ["mediastinum", "mediastinal", "hilar", "hilum"],
        "pleura": ["pleura", "pleural", "costophrenic", "pneumothorax", "effusion"],
    },
    "deepeyenet": {
        "superior retina": ["superior retina", "superior"],
        "inferior retina": ["inferior retina", "inferior"],
        "nasal retina": ["nasal retina", "nasal"],
        "temporal retina": ["temporal retina", "temporal"],
        "macula": ["macula", "macular", "fovea"],
        "optic disc": ["optic disc", "disc", "papilla"],
        "retinal vessels": ["retinal vessels", "vessel", "vascular", "artery", "vein"],
    },
}

CONCEPT_ANATOMY_PRIORS = {
    "cardiomegaly": ["cardiac silhouette"],
    "effusion": ["pleura", "left lower lung", "right lower lung"],
    "pleural effusion": ["pleura", "left lower lung", "right lower lung"],
    "pneumothorax": ["pleura", "left upper lung", "right upper lung"],
    "opacity": ["left upper lung", "left lower lung", "right upper lung", "right lower lung"],
    "pneumonia": ["left upper lung", "left lower lung", "right upper lung", "right lower lung"],
    "atelectasis": ["left lower lung", "right lower lung"],
    "edema": ["left upper lung", "left lower lung", "right upper lung", "right lower lung"],
    "fracture": ["pleura"],
    "macular hole": ["macula"],
    "cone dystrophy": ["macula"],
    "morning glory syndrome": ["optic disc"],
    "neovascularization of the disc": ["optic disc"],
    "uveitis": ["retina"],
}


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


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text)) if s.strip()]


def _sentence_status(sentence: str, concept: str) -> str:
    sent = sentence.lower()
    concept_l = normalize_concept(concept)
    if concept_l == "no acute cardiopulmonary abnormality" and re.search(r"\bno acute\b|\bno active disease\b|\bnormal chest\b", sent):
        return "normal"
    concept_pos = sent.find(concept_l) if concept_l else -1
    window = sent if concept_pos < 0 else sent[max(0, concept_pos - 48) : concept_pos + len(concept_l) + 48]
    if re.search(r"\bno\b.{0,160}\b(identified|seen|noted|present|evident|visualized)\b", sent):
        return "absent"
    if any(re.search(pattern, window) for pattern in NEGATION_PATTERNS):
        return "absent"
    if any(re.search(pattern, sent) for pattern in NORMAL_PATTERNS) and concept_l in {"no acute cardiopulmonary abnormality"}:
        return "normal"
    return "present"


def _relation_type(status: str) -> str:
    if status == "absent":
        return "has_absent_finding"
    if status == "normal":
        return "has_normal_status"
    return "has_present_finding"


def _find_sentence_anatomy(sentence: str, dataset: str) -> list[str]:
    dataset = normalize_dataset_name(dataset)
    sent = sentence.lower()
    anatomy = []
    for node, aliases in ANATOMY_ALIASES.get(dataset, {}).items():
        if any(re.search(rf"\b{re.escape(alias)}\b", sent) for alias in aliases):
            anatomy.append(node)
    return sorted(set(anatomy))


def _infer_anatomy_for_concept(concept: str, dataset: str) -> list[str]:
    dataset = normalize_dataset_name(dataset)
    anatomy_names = get_anatomy_names(dataset)
    concept_l = normalize_concept(concept)
    priors = []
    for key, nodes in CONCEPT_ANATOMY_PRIORS.items():
        if key in concept_l or concept_l in key:
            priors.extend(n for n in nodes if n in anatomy_names)
    if priors:
        return sorted(set(priors))
    if dataset == "iuxray":
        return [n for n in anatomy_names if "lung" in n]
    return anatomy_names[:]


def rule_extract_relations(record: dict[str, Any], concepts: list[str], dataset: str) -> list[dict[str, str]]:
    text = " ".join(str(record.get(field, "")) for field in ["clinical_description", "report_text"])
    sentences = _split_sentences(text)
    concepts_l = [(concept, normalize_concept(concept)) for concept in concepts]
    relations = []
    for sent in sentences:
        sent_l = sent.lower()
        present_concepts = [concept for concept, norm in concepts_l if norm and re.search(rf"\b{re.escape(norm)}\b", sent_l)]
        if not present_concepts:
            continue
        anatomy_nodes = _find_sentence_anatomy(sent, dataset)
        for concept in present_concepts:
            status = _sentence_status(sent, concept)
            nodes = anatomy_nodes or _infer_anatomy_for_concept(concept, dataset)
            for anatomy in nodes:
                relations.append(
                    {
                        "source": anatomy,
                        "type": _relation_type(status),
                        "target": concept,
                        "status": status,
                        "evidence": sent,
                        "extractor": "rules",
                    }
                )
    return relations


def llm_extract_relations(
    record: dict[str, Any],
    concepts: list[str],
    anatomy_names: list[str],
    cache: dict[str, Any],
    model: str = "gpt-4o-mini",
) -> list[dict[str, str]]:
    key = _record_key(record)
    if key in cache:
        cached = cache[key]
        return cached if isinstance(cached, list) else []
    if not os.environ.get("OPENAI_API_KEY"):
        return []
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = {
            "task": "Extract clinically grounded anatomy-finding relations from this report.",
            "allowed_anatomy_nodes": anatomy_names,
            "allowed_finding_nodes": concepts,
            "allowed_relation_types": ["has_present_finding", "has_absent_finding", "has_normal_status", "related_to"],
            "output_schema": [
                {"source": "anatomy node", "type": "relation type", "target": "finding node", "status": "present|absent|normal|unknown", "evidence": "short report span"}
            ],
            "report": str(record.get("report_text", "")),
            "clinical_description": str(record.get("clinical_description", "")),
        }
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You extract conservative medical graph relations. Use only allowed node names. Return only JSON array.",
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            temperature=0,
        )
        text = response.choices[0].message.content or "[]"
        parsed = json.loads(text[text.find("[") : text.rfind("]") + 1])
        allowed_anatomy = set(anatomy_names)
        allowed_concepts = set(concepts)
        allowed_types = {"has_present_finding", "has_absent_finding", "has_normal_status", "related_to"}
        relations = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            src = str(item.get("source", ""))
            tgt = str(item.get("target", ""))
            rel_type = str(item.get("type", "related_to"))
            if src in allowed_anatomy and tgt in allowed_concepts and rel_type in allowed_types:
                status = str(item.get("status", "unknown"))
                relations.append(
                    {
                        "source": src,
                        "type": rel_type,
                        "target": tgt,
                        "status": status,
                        "evidence": str(item.get("evidence", ""))[:300],
                        "extractor": "llm",
                    }
                )
        cache[key] = relations
        return relations
    except Exception:
        return []


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
    relation_extractor: str = "rules",
    relation_extractor_model: str = "gpt-4o-mini",
    relation_cache: str | Path | None = None,
    dataset: str = "iuxray",
) -> dict[str, Any]:
    counter: Counter[str] = Counter()
    relation_counter: Counter[tuple[str, str, str]] = Counter()
    relation_examples: dict[tuple[str, str, str], str] = {}
    per_record_relations: dict[str, list[dict[str, str]]] = {}
    per_record: dict[str, list[str]] = {}
    dataset = normalize_dataset_name(dataset)
    anatomy_names = get_anatomy_names(dataset)

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
    relation_cache_path = Path(relation_cache) if relation_cache else None
    if relation_cache_path and relation_cache_path.exists():
        relation_cache_data = json.loads(relation_cache_path.read_text())
    else:
        relation_cache_data = {}
    if relation_extractor != "none":
        for record in records:
            key = _record_key(record)
            record_concepts = [c for c in per_record.get(key, []) if c in concept_set]
            if not record_concepts:
                continue
            llm_relations = []
            if relation_extractor == "llm":
                llm_relations = llm_extract_relations(record, record_concepts, anatomy_names, relation_cache_data, model=relation_extractor_model)
            relations = llm_relations or rule_extract_relations(record, record_concepts, dataset)
            filtered = []
            for rel in relations:
                src = str(rel.get("source", ""))
                rel_type = str(rel.get("type", "related_to"))
                tgt = rule_normalize_concept(str(rel.get("target", "")))
                if src not in anatomy_names or tgt not in concept_set:
                    continue
                edge = (src, rel_type, tgt)
                relation_counter[edge] += 1
                if edge not in relation_examples and rel.get("evidence"):
                    relation_examples[edge] = str(rel.get("evidence", ""))
                filtered.append({**rel, "target": tgt})
            per_record_relations[key] = filtered
    if relation_cache_path:
        relation_cache_path.parent.mkdir(parents=True, exist_ok=True)
        relation_cache_path.write_text(json.dumps(relation_cache_data, indent=2, ensure_ascii=False))

    relations_out = [
        {"source": src, "type": rel, "target": tgt, "count": count, "evidence": relation_examples.get((src, rel, tgt), "")}
        for (src, rel, tgt), count in relation_counter.most_common()
        if (src in concept_set or src in anatomy_names) and tgt in concept_set
    ]
    return {
        "concepts": concepts,
        "anatomy_nodes": anatomy_names,
        "relations": relations_out,
        "per_record_concepts": per_record,
        "per_record_relations": per_record_relations,
        "source": "radgraph" if rad_docs else "keywords_or_lexicon",
        "normalizer": normalizer,
        "relation_extractor": relation_extractor,
    }
