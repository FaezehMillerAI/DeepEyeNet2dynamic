from __future__ import annotations

import argparse
import functools
import shutil
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import HFMedicalReportDataset, MedicalReportDataset, anatomy_prior_matrix, collate_fn, collate_hf_fn, get_anatomy_names
from .metrics import concept_metrics, graph_metrics, language_metrics
from .model import DynamicGraphCaptioner, GraphPrefixLLMCaptioner, GraphSeq2SeqCaptioner
from .utils import ensure_dir, get_device, load_json, save_json
from .visualize import (
    plot_concept_confusion,
    plot_counterfactual_curve,
    plot_dynamic_graph,
    plot_evidence_heatmap,
    plot_metric_bars,
    write_interactive_explanations,
)
from .vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["deepeyenet", "iuxray"], default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/deepeyenet_dynamic_graph/eval")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-report-len", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-counterfactuals", action="store_true")
    parser.add_argument("--max-interactive-examples", type=int, default=None)
    return parser.parse_args()


def _uses_hf_decoder(cfg: Config) -> bool:
    return cfg.decoder_type in {"llm", "causal_lm", "seq2seq"}


def _is_seq2seq_decoder(cfg: Config) -> bool:
    return cfg.decoder_type == "seq2seq"


def _prepare_tokenizer(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None and tokenizer.pad_token is not None:
        tokenizer.eos_token = tokenizer.pad_token
    return tokenizer


def _token_ids(tokenizer) -> tuple[int, int, int]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if bos_id is None:
        bos_id = pad_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    return int(pad_id), int(bos_id), int(eos_id)


def _build_hf_model(cfg: Config, tokenizer, concepts: list[str]):
    pad_id, bos_id, eos_id = _token_ids(tokenizer)
    model_cls = GraphSeq2SeqCaptioner if _is_seq2seq_decoder(cfg) else GraphPrefixLLMCaptioner
    return model_cls(
        cfg.llm_name,
        concepts,
        pad_id,
        bos_id,
        eos_id,
        cfg.embed_dim,
        cfg.hidden_dim,
        cfg.patch_grid,
        cfg.dropout,
        cfg.graph_steps,
        get_anatomy_names(cfg.dataset),
        anatomy_prior_matrix(cfg.dataset, cfg.patch_grid),
        cfg.use_anatomy,
        cfg.freeze_llm,
        cfg.prefix_length,
    )


def _top_region_ids(rc_edges: torch.Tensor, concept_ids: torch.Tensor) -> torch.Tensor:
    ids = []
    for b in range(rc_edges.shape[0]):
        concept_id = int(concept_ids[b])
        region_scores = rc_edges[b, :, :, concept_id].mean(dim=0)
        ids.append(int(region_scores.argmax()))
    return torch.tensor(ids, device=rc_edges.device, dtype=torch.long)


def _mask_regions(images: torch.Tensor, region_ids: torch.Tensor, patch_grid: int) -> torch.Tensor:
    masked = images.clone()
    batch, _, height, width = images.shape
    patch_h = height // patch_grid
    patch_w = width // patch_grid
    for b in range(batch):
        region_id = int(region_ids[b])
        row, col = divmod(region_id, patch_grid)
        y0, y1 = row * patch_h, (row + 1) * patch_h
        x0, x1 = col * patch_w, (col + 1) * patch_w
        masked[b, :, y0:y1, x0:x1] = 0.0
    return masked


def _top_anatomy_ids(region_anatomy_edges: torch.Tensor | None, region_ids: torch.Tensor) -> torch.Tensor | None:
    if region_anatomy_edges is None:
        return None
    ids = []
    for b in range(region_anatomy_edges.shape[0]):
        ids.append(int(region_anatomy_edges[b, int(region_ids[b])].argmax()))
    return torch.tensor(ids, device=region_anatomy_edges.device, dtype=torch.long)


def _decode_text(decoder, ids: list[int]) -> str:
    if hasattr(decoder, "itos"):
        return decoder.decode(ids)
    return decoder.decode(ids, skip_special_tokens=True)


def _split_sentences(text: str) -> list[str]:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(text).strip()) if s.strip()]
    return parts or [str(text).strip() or "empty generated report"]


def _linked_sentence_id(sentences: list[str], concept_names: list[str]) -> int:
    lowered = [s.lower() for s in sentences]
    for concept in concept_names:
        concept_l = concept.lower()
        for idx, sent in enumerate(lowered):
            if concept_l and concept_l in sent:
                return idx
    return 0


@torch.no_grad()
def evaluate_model(model, loader, text_decoder, concepts: list[str], cfg: Config, device: torch.device, data_root: Path, output_dir: Path) -> dict:
    model.eval()
    references, hypotheses = [], []
    all_true, all_prob = [], []
    all_rc, all_tc = [], []
    temporal_drifts = []
    patch_drops = []
    anatomy_drops = []
    finding_drops = []
    examples = []

    for batch in tqdm(loader, desc="evaluate"):
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        output, gen_tokens = model.generate(images, max_len=cfg.max_report_len)
        if attention_mask is not None:
            teacher_output = model(images, tokens, attention_mask=attention_mask)
        else:
            teacher_output = model(images, tokens)
        pred_texts = [_decode_text(text_decoder, row.tolist()) for row in gen_tokens.cpu()]
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
        region_ids = _top_region_ids(teacher_output.rc_edges, concept_ids)
        anatomy_ids = _top_anatomy_ids(teacher_output.region_anatomy_edges, region_ids)
        if not cfg.disable_counterfactuals:
            masked = _mask_regions(images, region_ids, cfg.patch_grid)
            masked_output = model(masked, tokens, attention_mask=attention_mask) if attention_mask is not None else model(masked, tokens)
            masked_probs = torch.sigmoid(masked_output.concept_logits)
            drops = probs.gather(1, concept_ids[:, None]) - masked_probs.gather(1, concept_ids[:, None])
            patch_drops.extend(drops.squeeze(1).cpu().tolist())

            if anatomy_ids is not None:
                anatomy_output = model(images, tokens, attention_mask=attention_mask, suppress_anatomy_ids=anatomy_ids) if attention_mask is not None else model(images, tokens, suppress_anatomy_ids=anatomy_ids)
                anatomy_probs = torch.sigmoid(anatomy_output.concept_logits)
                drops = probs.gather(1, concept_ids[:, None]) - anatomy_probs.gather(1, concept_ids[:, None])
                anatomy_drops.extend(drops.squeeze(1).cpu().tolist())

            finding_output = model(images, tokens, attention_mask=attention_mask, suppress_concept_ids=concept_ids) if attention_mask is not None else model(images, tokens, suppress_concept_ids=concept_ids)
            finding_probs = torch.sigmoid(finding_output.concept_logits)
            drops = probs.gather(1, concept_ids[:, None]) - finding_probs.gather(1, concept_ids[:, None])
            finding_drops.extend(drops.squeeze(1).cpu().tolist())

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
                    "region_anatomy_edges": None if teacher_output.region_anatomy_edges is None else teacher_output.region_anatomy_edges[i].cpu().numpy(),
                    "patch_cf_drop": float(patch_drops[-images.shape[0] + i]) if patch_drops and len(patch_drops) >= images.shape[0] else 0.0,
                    "anatomy_cf_drop": float(anatomy_drops[-images.shape[0] + i]) if anatomy_drops and len(anatomy_drops) >= images.shape[0] else 0.0,
                    "finding_cf_drop": float(finding_drops[-images.shape[0] + i]) if finding_drops and len(finding_drops) >= images.shape[0] else 0.0,
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
    metrics["patch_counterfactual_drop_mean"] = float(np.mean(patch_drops)) if patch_drops else 0.0
    metrics["patch_counterfactual_drop_median"] = float(np.median(patch_drops)) if patch_drops else 0.0
    metrics["patch_counterfactual_positive_rate"] = float(np.mean(np.asarray(patch_drops) > 0)) if patch_drops else 0.0
    metrics["anatomy_counterfactual_drop_mean"] = float(np.mean(anatomy_drops)) if anatomy_drops else 0.0
    metrics["anatomy_counterfactual_positive_rate"] = float(np.mean(np.asarray(anatomy_drops) > 0)) if anatomy_drops else 0.0
    metrics["finding_counterfactual_drop_mean"] = float(np.mean(finding_drops)) if finding_drops else 0.0
    metrics["finding_counterfactual_positive_rate"] = float(np.mean(np.asarray(finding_drops) > 0)) if finding_drops else 0.0

    save_json(metrics, output_dir / "metrics.json")
    save_json(
        [{"reference": r, "prediction": h} for r, h in zip(references, hypotheses)],
        output_dir / "generated_reports.json",
    )
    plot_metric_bars(metrics, output_dir / "metric_summary.png")
    plot_concept_confusion(y_true, y_prob, concepts, output_dir / "concept_confusion_top20.png")
    plot_counterfactual_curve(patch_drops, output_dir / "counterfactual_evidence_drop.png")

    interactive_examples = []
    interactive_image_dir = output_dir / "interactive_images"
    interactive_image_dir.mkdir(parents=True, exist_ok=True)
    anatomy_names = getattr(model, "anatomy_names", [])
    for idx, ex in enumerate(examples):
        concept_scores = ex["concept_prob"]
        top_concept = int(np.argmax(concept_scores))
        region_scores = ex["rc_edges"][:, :, top_concept].mean(axis=0)
        full_image_path = data_root / ex["image_path"]
        if full_image_path.exists():
            plot_evidence_heatmap(full_image_path, region_scores, cfg.patch_grid, output_dir / f"example_{idx}_evidence_heatmap.png")
        plot_dynamic_graph(region_scores, concept_scores, concepts, output_dir / f"example_{idx}_dynamic_graph.png")
        if idx < cfg.max_interactive_examples and full_image_path.exists():
            image_copy_name = f"case_{idx}{full_image_path.suffix.lower() or '.png'}"
            image_copy_path = interactive_image_dir / image_copy_name
            if not image_copy_path.exists():
                shutil.copy2(full_image_path, image_copy_path)
            rc_mean = ex["rc_edges"].mean(axis=0)
            report_sentences = _split_sentences(ex["prediction"])
            region_strength = region_scores - region_scores.min()
            region_strength = region_strength / (region_strength.max() + 1e-8)
            patches = []
            for patch_id in range(cfg.patch_grid * cfg.patch_grid):
                top_concepts = np.argsort(-rc_mean[patch_id])[: min(3, len(concepts))]
                top_concept_names = [concepts[int(c)] for c in top_concepts]
                linked_sid = _linked_sentence_id(report_sentences, top_concept_names)
                anatomy = "region"
                if ex["region_anatomy_edges"] is not None and anatomy_names:
                    anatomy_id = int(np.argmax(ex["region_anatomy_edges"][patch_id]))
                    anatomy = anatomy_names[anatomy_id]
                patches.append(
                    {
                        "patch_id": patch_id,
                        "anatomy": anatomy,
                        "top_concepts": [{"name": concepts[int(c)], "score": float(rc_mean[patch_id, int(c)])} for c in top_concepts],
                        "evidence_score": float(region_strength[patch_id]),
                        "linked_sentence_id": int(linked_sid),
                        "linked_report_text": report_sentences[linked_sid],
                        "patch_counterfactual_drop": ex["patch_cf_drop"] if patch_id == int(np.argmax(region_scores)) else 0.0,
                        "anatomy_counterfactual_drop": ex["anatomy_cf_drop"] if patch_id == int(np.argmax(region_scores)) else 0.0,
                        "finding_counterfactual_drop": ex["finding_cf_drop"] if patch_id == int(np.argmax(region_scores)) else 0.0,
                    }
                )
            interactive_examples.append(
                {
                    "image_path": ex["image_path"],
                    "image_src": f"interactive_images/{image_copy_name}",
                    "patch_grid": cfg.patch_grid,
                    "reference": ex["reference"],
                    "prediction": ex["prediction"],
                    "report_sentences": report_sentences,
                    "keywords": ex["keywords"],
                    "patches": patches,
                }
            )
    save_json(interactive_examples, output_dir / "interactive_explanations.json")
    write_interactive_explanations(interactive_examples, output_dir / "interactive_explanations.html")

    return metrics


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent
    cfg_path = run_dir / "config.json"
    cfg = Config.load(cfg_path) if cfg_path.exists() else Config(data_root=args.data_root)
    if not (run_dir / "tokenizer_config.json").exists() and not (run_dir / "tokenizer.json").exists() and (run_dir / "vocab.json").exists():
        cfg.decoder_type = "gru"
    cfg.data_root = args.data_root
    if args.dataset is not None:
        cfg.dataset = args.dataset
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.device = args.device
    if args.max_report_len is not None:
        cfg.max_report_len = args.max_report_len
    if args.no_counterfactuals:
        cfg.disable_counterfactuals = True
    if args.max_interactive_examples is not None:
        cfg.max_interactive_examples = args.max_interactive_examples
    output_dir = ensure_dir(args.output_dir)
    device = get_device(cfg.device)

    concepts = load_json(run_dir / "concepts.json")["concepts"]
    if _uses_hf_decoder(cfg):
        from transformers import AutoTokenizer

        tokenizer_source = run_dir if (run_dir / "tokenizer_config.json").exists() else cfg.llm_name
        tokenizer = _prepare_tokenizer(AutoTokenizer.from_pretrained(tokenizer_source))
        dataset = HFMedicalReportDataset(cfg.data_root, args.split, tokenizer, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
        pad_id, _, _ = _token_ids(tokenizer)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=functools.partial(collate_hf_fn, pad_id=pad_id))
        model = _build_hf_model(cfg, tokenizer, concepts).to(device)
        text_decoder = tokenizer
    else:
        vocab = Vocabulary.from_dict(load_json(run_dir / "vocab.json"))
        dataset = MedicalReportDataset(cfg.data_root, args.split, vocab, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
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
            get_anatomy_names(cfg.dataset),
            anatomy_prior_matrix(cfg.dataset, cfg.patch_grid),
            cfg.use_anatomy,
        ).to(device)
        text_decoder = vocab
    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"Loaded checkpoint with missing={len(missing)} unexpected={len(unexpected)} keys. This is expected when evaluating older checkpoints after architecture upgrades.")
    metrics = evaluate_model(model, loader, text_decoder, concepts, cfg, device, Path(cfg.data_root), output_dir)
    print(metrics)
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
