from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from PIL import Image
from sklearn.metrics import multilabel_confusion_matrix


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 130


def plot_metric_bars(metrics: dict[str, float], output_path: str | Path) -> None:
    set_style()
    items = [(k, v) for k, v in metrics.items() if isinstance(v, (int, float)) and np.isfinite(v)]
    labels, values = zip(*items) if items else ([], [])
    plt.figure(figsize=(max(9, len(labels) * 0.45), 5))
    ax = sns.barplot(x=list(labels), y=list(values), hue=list(labels), palette="viridis", legend=False)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_concept_confusion(y_true: np.ndarray, y_prob: np.ndarray, concepts: Sequence[str], output_path: str | Path, top_n: int = 20) -> None:
    set_style()
    y_pred = (y_prob >= 0.5).astype(int)
    support = y_true.sum(axis=0)
    ids = np.argsort(-support)[: min(top_n, len(concepts))]
    cms = multilabel_confusion_matrix(y_true[:, ids], y_pred[:, ids])
    fns = cms[:, 1, 0]
    fps = cms[:, 0, 1]
    tps = cms[:, 1, 1]
    mat = np.vstack([tps, fps, fns])
    plt.figure(figsize=(max(9, len(ids) * 0.55), 4.5))
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="mako", xticklabels=[concepts[i] for i in ids], yticklabels=["TP", "FP", "FN"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_evidence_heatmap(image_path: str | Path, region_scores: np.ndarray, patch_grid: int, output_path: str | Path) -> None:
    set_style()
    image = Image.open(image_path).convert("RGB")
    scores = region_scores.reshape(patch_grid, patch_grid)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.imshow(scores, cmap="inferno", alpha=0.45, extent=(0, image.width, image.height, 0), interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_dynamic_graph(
    region_scores: np.ndarray,
    concept_scores: np.ndarray,
    concepts: Sequence[str],
    output_path: str | Path,
    top_regions: int = 5,
    top_concepts: int = 6,
) -> None:
    set_style()
    r_ids = np.argsort(-region_scores)[:top_regions]
    c_ids = np.argsort(-concept_scores)[:top_concepts]
    graph = nx.DiGraph()
    for r in r_ids:
        graph.add_node(f"R{r}", kind="region")
    for c in c_ids:
        graph.add_node(concepts[c], kind="concept")
    for r in r_ids:
        for c in c_ids[:3]:
            graph.add_edge(f"R{r}", concepts[c], weight=float(region_scores[r] * concept_scores[c]))

    pos = {}
    for i, r in enumerate(r_ids):
        pos[f"R{r}"] = (0, -i)
    for i, c in enumerate(c_ids):
        pos[concepts[c]] = (1.8, -i)
    colors = ["#4C78A8" if graph.nodes[n]["kind"] == "region" else "#F58518" for n in graph.nodes]
    widths = [1 + 4 * graph.edges[e]["weight"] / max(1e-8, max(nx.get_edge_attributes(graph, "weight").values())) for e in graph.edges]
    plt.figure(figsize=(9, 6))
    nx.draw_networkx(graph, pos=pos, node_color=colors, width=widths, arrows=True, font_size=9, node_size=1400)
    plt.axis("off")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def plot_counterfactual_curve(drops: Sequence[float], output_path: str | Path) -> None:
    set_style()
    plt.figure(figsize=(7, 4.5))
    sns.histplot(drops, bins=15, kde=True, color="#2A9D8F")
    plt.xlabel("Predicted concept confidence drop after top-evidence masking")
    plt.ylabel("Number of images")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
