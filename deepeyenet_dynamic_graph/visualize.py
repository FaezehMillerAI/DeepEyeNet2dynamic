from __future__ import annotations

import os
import tempfile
import json
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
    if values:
        ymin = min(0.0, min(values) * 1.15)
        ymax = max(1.0, max(values) * 1.15)
        ax.set_ylim(ymin, ymax)
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


def write_interactive_explanations(examples: list[dict], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(examples)
    page = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Interactive Explanation Graph Viewer</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2933; background: #f7f8fa; }}
.case {{ display: grid; grid-template-columns: minmax(320px, 520px) 1fr; gap: 24px; margin-bottom: 36px; padding: 18px; background: white; border: 1px solid #d8dee9; border-radius: 8px; }}
.image-wrap {{ position: relative; width: 100%; max-width: 520px; }}
.image-wrap img {{ width: 100%; display: block; border-radius: 6px; }}
.patch {{ position: absolute; border: 1px solid rgba(255,255,255,.42); background: rgba(42,157,143,.08); cursor: crosshair; }}
.patch:hover {{ background: rgba(244,162,97,.28); outline: 2px solid #f4a261; z-index: 2; }}
.tooltip {{ position: fixed; display: none; max-width: 360px; padding: 12px; background: #111827; color: white; border-radius: 6px; font-size: 13px; line-height: 1.35; pointer-events: none; z-index: 99; box-shadow: 0 12px 28px rgba(0,0,0,.28); }}
.report {{ white-space: pre-wrap; line-height: 1.5; }}
.meta {{ color: #52606d; font-size: 13px; }}
.pill {{ display: inline-block; padding: 2px 6px; border-radius: 999px; background: #e6f4f1; margin: 2px 4px 2px 0; font-size: 12px; }}
h1 {{ margin-top: 0; }}
h2 {{ margin: 0 0 8px; }}
</style>
</head>
<body>
<h1>Interactive Explanation Graph Viewer</h1>
<p class="meta">Hover over image patches to inspect anatomy, top findings, linked report text, and counterfactual evidence drops.</p>
<div id="root"></div>
<div id="tooltip" class="tooltip"></div>
<script>
const examples = {payload};
const root = document.getElementById('root');
const tooltip = document.getElementById('tooltip');
function esc(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}}[c])); }}
function patchTip(p) {{
  const concepts = p.top_concepts.map(c => `<span class="pill">${{esc(c.name)}} ${{Number(c.score).toFixed(3)}}</span>`).join('');
  return `<b>Patch R${{p.patch_id}}</b><br>
  Anatomy: <b>${{esc(p.anatomy)}}</b><br>
  Top findings:<br>${{concepts}}<br>
  Linked report: <i>${{esc(p.linked_report_text)}}</i><br>
  Patch CF drop: ${{Number(p.patch_counterfactual_drop ?? 0).toFixed(3)}}<br>
  Anatomy CF drop: ${{Number(p.anatomy_counterfactual_drop ?? 0).toFixed(3)}}<br>
  Finding CF drop: ${{Number(p.finding_counterfactual_drop ?? 0).toFixed(3)}}`;
}}
examples.forEach((ex, idx) => {{
  const div = document.createElement('section');
  div.className = 'case';
  const image = document.createElement('div');
  image.className = 'image-wrap';
  image.innerHTML = `<img src="${{esc(ex.image_src)}}" alt="case ${{idx}}">`;
  const grid = ex.patch_grid;
  ex.patches.forEach(p => {{
    const cell = document.createElement('div');
    cell.className = 'patch';
    const row = Math.floor(p.patch_id / grid);
    const col = p.patch_id % grid;
    cell.style.left = `${{100 * col / grid}}%`;
    cell.style.top = `${{100 * row / grid}}%`;
    cell.style.width = `${{100 / grid}}%`;
    cell.style.height = `${{100 / grid}}%`;
    cell.addEventListener('mousemove', ev => {{
      tooltip.style.display = 'block';
      tooltip.style.left = (ev.clientX + 14) + 'px';
      tooltip.style.top = (ev.clientY + 14) + 'px';
      tooltip.innerHTML = patchTip(p);
    }});
    cell.addEventListener('mouseleave', () => tooltip.style.display = 'none');
    image.appendChild(cell);
  }});
  const text = document.createElement('div');
  text.innerHTML = `<h2>${{esc(ex.image_path)}}</h2>
    <div class="meta">Keywords: ${{ex.keywords.map(esc).join(', ')}}</div>
    <h3>Generated Report</h3><div class="report">${{esc(ex.prediction)}}</div>
    <h3>Reference Report</h3><div class="report">${{esc(ex.reference)}}</div>`;
  div.appendChild(image);
  div.appendChild(text);
  root.appendChild(div);
}});
</script>
</body>
</html>"""
    output_path.write_text(page, encoding="utf-8")
