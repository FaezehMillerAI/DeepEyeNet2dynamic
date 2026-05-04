from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from .vocab import tokenize


def language_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    try:
        from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

        refs = [[tokenize(r)] for r in references]
        hyps = [tokenize(h) for h in hypotheses]
        smooth = SmoothingFunction().method1
        for n in range(1, 5):
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            scores[f"bleu_{n}"] = float(corpus_bleu(refs, hyps, weights=weights, smoothing_function=smooth))
    except Exception:
        for n in range(1, 5):
            scores[f"bleu_{n}"] = float("nan")

    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        vals = [scorer.score(r, h)["rougeL"].fmeasure for r, h in zip(references, hypotheses)]
        scores["rouge_l"] = float(np.mean(vals))
    except Exception:
        scores["rouge_l"] = float("nan")

    try:
        from nltk.translate.meteor_score import meteor_score

        scores["meteor"] = float(np.mean([meteor_score([tokenize(r)], tokenize(h)) for r, h in zip(references, hypotheses)]))
    except Exception:
        scores["meteor"] = float("nan")
    return scores


def concept_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    micro = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    exact = (y_true == y_pred).all(axis=1).mean() if len(y_true) else 0.0
    return {
        "concept_precision_micro": float(micro[0]),
        "concept_recall_micro": float(micro[1]),
        "concept_f1_micro": float(micro[2]),
        "concept_precision_macro": float(macro[0]),
        "concept_recall_macro": float(macro[1]),
        "concept_f1_macro": float(macro[2]),
        "concept_exact_match": float(exact),
    }


def graph_metrics(rc_edges: np.ndarray, token_concept_edges: np.ndarray, y_true: np.ndarray, topk: int = 3) -> dict[str, float]:
    eps = 1e-8
    edge_entropy = float(-(token_concept_edges * np.log(token_concept_edges + eps)).sum(axis=-1).mean())
    rc_entropy = float(-(rc_edges * np.log(rc_edges + eps)).sum(axis=-1).mean())
    temp_drift = 0.0
    if rc_edges.shape[1] > 1:
        temp_drift = float(np.abs(rc_edges[:, 1:] - rc_edges[:, :-1]).mean())

    top_hits = []
    mean_token_concept = token_concept_edges.mean(axis=1)
    for probs, truth in zip(mean_token_concept, y_true):
        true_ids = set(np.where(truth > 0)[0].tolist())
        if not true_ids:
            continue
        pred_ids = set(np.argsort(-probs)[:topk].tolist())
        top_hits.append(len(true_ids & pred_ids) / min(len(true_ids), topk))
    return {
        "token_concept_entropy": edge_entropy,
        "region_concept_entropy": rc_entropy,
        "temporal_graph_drift": temp_drift,
        f"top{topk}_concept_hit_rate": float(np.mean(top_hits)) if top_hits else 0.0,
    }
