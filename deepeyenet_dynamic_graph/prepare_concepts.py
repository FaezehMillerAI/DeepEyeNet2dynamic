from __future__ import annotations

import argparse
from pathlib import Path

from .concept_graph import build_concept_graph
from .data import load_split_records
from .utils import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build RadGraph/hybrid clinical concept graph artifacts.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["deepeyenet", "iuxray"], default="iuxray")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--output", default="outputs/concept_graph.json")
    parser.add_argument("--max-concepts", type=int, default=128)
    parser.add_argument("--radgraph-path", default=None)
    parser.add_argument("--concept-normalizer", choices=["rules", "llm"], default="rules")
    parser.add_argument("--concept-normalizer-model", default="gpt-4o-mini")
    parser.add_argument("--relation-extractor", choices=["none", "rules", "llm"], default="rules")
    parser.add_argument("--relation-extractor-model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_split_records(args.data_root, args.split, dataset=args.dataset, seed=args.seed)
    output = Path(args.output)
    graph = build_concept_graph(
        records,
        args.max_concepts,
        radgraph_path=args.radgraph_path,
        normalizer=args.concept_normalizer,
        normalizer_cache=output.parent / "concept_normalization_cache.json",
        llm_model=args.concept_normalizer_model,
        relation_extractor=args.relation_extractor,
        relation_extractor_model=args.relation_extractor_model,
        relation_cache=output.parent / "relation_extraction_cache.json",
        dataset=args.dataset,
    )
    save_json(graph, output)
    print(f"Saved concept graph to {output}")
    print(f"Concepts: {len(graph['concepts'])}")
    print(f"Relations: {len(graph['relations'])}")
    print(f"Source: {graph['source']}")
    print("Top concepts:")
    for concept in graph["concepts"][:20]:
        print(f"- {concept}")
    print("Top relations:")
    for rel in graph["relations"][:20]:
        print(f"- {rel['source']} --{rel['type']}--> {rel['target']} ({rel['count']})")


if __name__ == "__main__":
    main()
