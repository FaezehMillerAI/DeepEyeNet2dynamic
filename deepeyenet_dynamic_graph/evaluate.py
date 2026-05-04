from __future__ import annotations

import argparse
import functools
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import DeepEyeNetDataset, collate_fn
from .metrics import concept_metrics, graph_metrics, language_metrics
from .model import DynamicGraphCaptioner
from .utils import ensure_dir, get_device, load_json, save_json
from .visualize import (
    plot_concept_confusion,
    plot_counterfactual_curve,
    plot_dynamic_graph,
    plot_evidence_heatmap,
    plot_metric_bars,
)
from .vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/deepeyenet_dynamic_graph/eval")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-report-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _mask_top_regions(images: torch.Tensor, rc_edges: torch.Tensor, concept_ids: torch.Tensor, patch_grid: int) -> torch.Tensor:
    masked = images.clone()
    batch, _, height, width = images.shape
    patch_h = height // patch_grid
    patch_w = width // patch_grid
    for b in range(batch):
        concept_id = int(concept_ids[b])
        region_scores = rc_edges[b, :, :, concept_id].mean(dim=0)
        region_id = int(region_scores.argmax())
        row, col = divmod(region_id, patch_grid)
        y0, y1 = row * patch_h, (row + 1) * patch_h
        x0, x1 = col * patch_w, (col + 1) * patch_w
        masked[b, :, y0:y1, x0:x1] = 0.0
    return masked


@torch.no_grad()
def evaluate_model(model, loader, vocab: Vocabulary, concepts: list[str], cfg: Config, device: torch.device, data_root: Path, output_dir: Path) -> dict:
    model.eval()
    references, hypotheses = [], []
    all_true, all_prob = [], []
    all_rc, all_tc = [], []
    temporal_drifts = []
    faithfulness_drops = []
    examples = []

    for batch in tqdm(loader, desc="evaluate"):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        output, gen_tokens = model.generate(images, max_len=cfg.max_report_len)
        teacher_output = model(images, tokens)
        pred_texts = [vocab.decode(row.tolist()) for row in gen_tokens.cpu()]
        references.extend(batch["report_text"])
        hypotheses.extend(pred_texts)

        probs = torch.sigmoid(teacher_output.concept_logits)
        all_true.append(batch["concept_targets"].numpy())
        all_prob.append(probs.cpu().numpy())
        all_rc.append(teacher_output.rc_edges.mean(dim=1).cpu().numpy())
        all_tc.append(teacher_output.token_concept_edges.mean(dim=1).cpu().numpy())
        if teacher_output.rc_edges.shape[1] > 1:
            drift = (teacher_output.rc_edges[:, 1:] - teacher_output.rc_edges[:, :-1]).abs().mean(dim=(1, 2, 3))
            temporal_drifts.extend(drift.cpu().tolist())

        concept_ids = probs.argmax(dim=1)
        masked = _mask_top_regions(images, teacher_output.rc_edges, concept_ids, cfg.patch_grid)
        masked_output = model(masked, tokens)
        masked_probs = torch.sigmoid(masked_output.concept_logits)
        drops = probs.gather(1, concept_ids[:, None]) - masked_probs.gather(1, concept_ids[:, None])
        faithfulness_drops.extend(drops.squeeze(1).cpu().tolist())

        for i in range(min(3, images.shape[0])):
            if len(examples) >= 6:
                break
            examples.append(
                {
                    "image_path": batch["image_path"][i],
                    "reference": batch["report_text"][i],
                    "prediction": pred_texts[i],
                    "keywords": batch["keywords"][i],
                    "concept_prob": probs[i].cpu().numpy(),
                    "rc_edges": teacher_output.rc_edges[i].cpu().numpy(),
                    "token_concept_edges": teacher_output.token_concept_edges[i].cpu().numpy(),
                }
            )

    y_true = np.concatenate(all_true, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)
    rc = np.concatenate(all_rc, axis=0)
    tc = np.concatenate(all_tc, axis=0)
    metrics = {}
    metrics.update(language_metrics(references, hypotheses))
    metrics.update(concept_metrics(y_true, y_prob))
    metrics.update(graph_metrics(rc, tc, y_true, topk=cfg.topk_evidence, temporal_drifts=np.asarray(temporal_drifts)))
    metrics["faithfulness_confidence_drop_mean"] = float(np.mean(faithfulness_drops)) if faithfulness_drops else 0.0
    metrics["faithfulness_confidence_drop_median"] = float(np.median(faithfulness_drops)) if faithfulness_drops else 0.0

    save_json(metrics, output_dir / "metrics.json")
    save_json(
        [{"reference": r, "prediction": h} for r, h in zip(references, hypotheses)],
        output_dir / "generated_reports.json",
    )
    plot_metric_bars(metrics, output_dir / "metric_summary.png")
    plot_concept_confusion(y_true, y_prob, concepts, output_dir / "concept_confusion_top20.png")
    plot_counterfactual_curve(faithfulness_drops, output_dir / "counterfactual_evidence_drop.png")

    for idx, ex in enumerate(examples):
        concept_scores = ex["concept_prob"]
        top_concept = int(np.argmax(concept_scores))
        region_scores = ex["rc_edges"][:, :, top_concept].mean(axis=0)
        full_image_path = data_root / ex["image_path"]
        if full_image_path.exists():
            plot_evidence_heatmap(full_image_path, region_scores, cfg.patch_grid, output_dir / f"example_{idx}_evidence_heatmap.png")
        plot_dynamic_graph(region_scores, concept_scores, concepts, output_dir / f"example_{idx}_dynamic_graph.png")

    return metrics


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    cfg_path = run_dir / "config.json"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config(data_root=args.data_root)
    cfg.data_root = args.data_root
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.device = args.device
    if args.max_report_len is not None:
        cfg.max_report_len = args.max_report_len
    output_dir = ensure_dir(args.output_dir)
    device = get_device(cfg.device)

    vocab = Vocabulary.from_dict(load_json(run_dir / "vocab.json"))
    concepts = load_json(run_dir / "concepts.json")["concepts"]
    dataset = DeepEyeNetDataset(cfg.data_root, args.split, vocab, concepts, cfg.image_size, cfg.max_report_len)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=functools.partial(collate_fn, pad_id=vocab.pad_id))

    model = DynamicGraphCaptioner(
        len(vocab.itos),
        concepts,
        vocab.pad_id,
        vocab.bos_id,
        vocab.eos_id,
        cfg.embed_dim,
        cfg.hidden_dim,
        cfg.patch_grid,
        cfg.dropout,
        cfg.graph_steps,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    metrics = evaluate_model(model, loader, vocab, concepts, cfg, device, Path(cfg.data_root), output_dir)
    print(metrics)
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
