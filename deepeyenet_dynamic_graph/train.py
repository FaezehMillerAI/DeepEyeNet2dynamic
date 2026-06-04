from __future__ import annotations

import argparse
import csv
import functools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .concept_graph import build_concept_graph
from .data import HFMedicalReportDataset, MedicalReportDataset, anatomy_prior_matrix, build_artifacts, collate_fn, collate_hf_fn, get_anatomy_names, load_split_records
from .model import DynamicGraphCaptioner, GraphPrefixLLMCaptioner, GraphSeq2SeqCaptioner, compute_losses
from .report_memory import save_report_memory
from .utils import ensure_dir, get_device, save_json, set_seed
from .vocab import build_concepts


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["deepeyenet", "iuxray", "mimic_cxr"], default="deepeyenet")
    parser.add_argument("--output-dir", default="outputs/deepeyenet_dynamic_graph")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--max-report-len", type=int, default=96)
    parser.add_argument("--max-concepts", type=int, default=128)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-valid-records", type=int, default=None)
    parser.add_argument("--concept-source", choices=["keywords", "hybrid", "radgraph"], default="hybrid")
    parser.add_argument("--radgraph-path", default=None)
    parser.add_argument("--concept-normalizer", choices=["rules", "llm"], default="rules")
    parser.add_argument("--concept-normalizer-model", default="gpt-4o-mini")
    parser.add_argument("--relation-extractor", choices=["none", "rules", "llm"], default="rules")
    parser.add_argument("--relation-extractor-model", default="gpt-4o-mini")
    parser.add_argument("--relation-prior-weight", type=float, default=1.0)
    parser.add_argument("--vision-encoder-type", choices=["cnn", "hf", "torchvision", "radimagenet", "biomedclip"], default="cnn")
    parser.add_argument("--vision-encoder-name", default=None)
    parser.add_argument("--vision-checkpoint", default=None)
    parser.add_argument("--freeze-vision-encoder", action="store_true")
    parser.add_argument("--decoder-type", choices=["llm", "causal_lm", "seq2seq", "gru"], default="llm")
    parser.add_argument("--llm-name", default="distilgpt2")
    parser.add_argument("--llm-trust-remote-code", action="store_true")
    parser.add_argument("--llm-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--llm-attn-implementation", default=None)
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--decoder-prompt", default=None)
    parser.add_argument("--prefix-length", type=int, default=4)
    parser.add_argument("--concept-logit-bias", type=float, default=0.8)
    parser.add_argument("--graph-steps", type=int, default=1)
    parser.add_argument("--lambda-concept", type=float, default=0.4)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--lambda-coverage", type=float, default=0.05)
    parser.add_argument("--lambda-sparse", type=float, default=0.01)
    parser.add_argument("--lambda-temp", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-anatomy", action="store_true")
    parser.add_argument("--disable-counterfactuals", action="store_true")
    parser.add_argument("--no-report-memory", action="store_true")
    parser.add_argument("--report-memory-max-entries", type=int, default=2500)
    parser.add_argument("--report-memory-min-score", type=float, default=0.20)
    parser.add_argument("--generation-num-beams", type=int, default=3)
    parser.add_argument("--generation-min-len", type=int, default=24)
    parser.add_argument("--generation-no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--generation-repetition-penalty", type=float, default=1.15)
    parser.add_argument("--generation-length-penalty", type=float, default=1.0)
    parser.add_argument("--decoder-concept-evidence-topk", type=int, default=12)
    parser.add_argument("--decoder-region-evidence-topk", type=int, default=8)
    parser.add_argument("--progress-style", choices=["epoch", "batch", "none"], default="epoch")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = Config(
        data_root=args.data_root,
        dataset=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        patch_grid=args.patch_grid,
        max_report_len=args.max_report_len,
        max_concepts=args.max_concepts,
        max_train_records=args.max_train_records,
        max_valid_records=args.max_valid_records,
        concept_source=args.concept_source,
        radgraph_path=args.radgraph_path,
        concept_normalizer=args.concept_normalizer,
        concept_normalizer_model=args.concept_normalizer_model,
        relation_extractor=args.relation_extractor,
        relation_extractor_model=args.relation_extractor_model,
        relation_prior_weight=args.relation_prior_weight,
        vision_encoder_type=args.vision_encoder_type,
        vision_encoder_name=args.vision_encoder_name,
        vision_checkpoint=args.vision_checkpoint,
        freeze_vision_encoder=args.freeze_vision_encoder,
        decoder_type=args.decoder_type,
        llm_name=args.llm_name,
        llm_trust_remote_code=args.llm_trust_remote_code,
        llm_dtype=args.llm_dtype,
        llm_attn_implementation=args.llm_attn_implementation,
        freeze_llm=args.freeze_llm,
        decoder_prompt=args.decoder_prompt,
        prefix_length=args.prefix_length,
        concept_logit_bias=args.concept_logit_bias,
        graph_steps=args.graph_steps,
        lambda_concept=args.lambda_concept,
        lambda_align=args.lambda_align,
        lambda_coverage=args.lambda_coverage,
        lambda_sparse=args.lambda_sparse,
        lambda_temp=args.lambda_temp,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        use_anatomy=not args.no_anatomy,
        disable_counterfactuals=args.disable_counterfactuals,
        use_report_memory=not args.no_report_memory,
        report_memory_max_entries=args.report_memory_max_entries,
        report_memory_min_score=args.report_memory_min_score,
        generation_num_beams=args.generation_num_beams,
        generation_min_len=args.generation_min_len,
        generation_no_repeat_ngram_size=args.generation_no_repeat_ngram_size,
        generation_repetition_penalty=args.generation_repetition_penalty,
        generation_length_penalty=args.generation_length_penalty,
        decoder_concept_evidence_topk=args.decoder_concept_evidence_topk,
        decoder_region_evidence_topk=args.decoder_region_evidence_topk,
        progress_style=args.progress_style,
        device=args.device,
    )
    return cfg


def _uses_hf_decoder(cfg: Config) -> bool:
    return cfg.decoder_type in {"llm", "causal_lm", "seq2seq"}


def _is_seq2seq_decoder(cfg: Config) -> bool:
    return cfg.decoder_type == "seq2seq"


def _limit_records(records: list[dict], max_records: int | None, label: str) -> list[dict]:
    if max_records is None or max_records <= 0 or len(records) <= max_records:
        return records
    tqdm.write(f"Using first {max_records:,} {label} records out of {len(records):,}.")
    return records[:max_records]


def _prepare_tokenizer(tokenizer):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token is None and tokenizer.pad_token is not None:
        tokenizer.eos_token = tokenizer.pad_token
    return tokenizer


def _tokenizer_kwargs(cfg: Config) -> dict:
    return {"trust_remote_code": bool(cfg.llm_trust_remote_code)}


def _token_ids(tokenizer) -> tuple[int, int, int]:
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if bos_id is None:
        bos_id = pad_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    return int(pad_id), int(bos_id), int(eos_id)


def _anatomy_concept_prior_from_graph(cfg: Config, concepts: list[str], concept_graph: dict | None) -> torch.Tensor:
    anatomy_names = get_anatomy_names(cfg.dataset)
    prior = torch.ones(len(anatomy_names), len(concepts), dtype=torch.float32)
    if concept_graph:
        anatomy_to_idx = {name: idx for idx, name in enumerate(anatomy_names)}
        concept_to_idx = {name: idx for idx, name in enumerate(concepts)}
        for rel in concept_graph.get("relations", []):
            src = rel.get("source")
            tgt = rel.get("target")
            if src in anatomy_to_idx and tgt in concept_to_idx:
                weight = float(rel.get("count", 1.0))
                if rel.get("type") == "has_present_finding":
                    weight *= 1.25
                prior[anatomy_to_idx[src], concept_to_idx[tgt]] += weight
    return prior / prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def _build_hf_model(cfg: Config, tokenizer, concepts: list[str], concept_graph: dict | None = None):
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
        _anatomy_concept_prior_from_graph(cfg, concepts, concept_graph),
        cfg.relation_prior_weight,
        cfg.use_anatomy,
        cfg.vision_encoder_type,
        cfg.vision_encoder_name,
        cfg.vision_checkpoint,
        cfg.freeze_vision_encoder,
        cfg.freeze_llm,
        cfg.prefix_length,
        cfg.concept_logit_bias,
        cfg.llm_trust_remote_code,
        cfg.llm_dtype,
        cfg.llm_attn_implementation,
        cfg.decoder_prompt,
    )


def _progress_postfix(totals: dict[str, float], n: int) -> dict[str, str]:
    keys = ["loss", "rep_loss", "concept_loss", "align_loss", "coverage_loss"]
    return {key.replace("_loss", ""): f"{totals[key] / max(1, n):.4f}" for key in keys if key in totals}


def _write_history_csv(history: list[dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    keys = ["epoch"]
    for row in history:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (out_dir / "training_progress.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def _plot_history(history: list[dict[str, float]], out_dir: Path) -> None:
    if not history:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    epochs = [row["epoch"] for row in history]
    panels = [
        ("Total Loss", ["train_loss", "valid_loss"]),
        ("Report Loss", ["train_rep_loss", "valid_rep_loss"]),
        ("Concept/Alignment", ["train_concept_loss", "valid_concept_loss", "train_align_loss", "valid_align_loss"]),
        ("Coverage/Sparsity", ["train_coverage_loss", "valid_coverage_loss", "train_sparse_loss", "valid_sparse_loss"]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (title, keys) in zip(axes.ravel(), panels):
        plotted = False
        for key in keys:
            vals = [row.get(key) for row in history]
            if all(v is None for v in vals):
                continue
            ax.plot(epochs, vals, marker="o", linewidth=2, label=key)
            plotted = True
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "training_progress.png", dpi=160)
    plt.close(fig)


def run_epoch(model, loader, optimizer, cfg: Config, device: torch.device, train: bool, epoch: int) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    n = 0
    stage = "train" if train else "valid"
    use_batch_bar = cfg.progress_style == "batch"
    iterator = tqdm(loader, desc=f"{stage} {epoch}/{cfg.epochs}", leave=False, dynamic_ncols=True, smoothing=0.05) if use_batch_bar else loader
    for batch in iterator:
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        concept_targets = batch["concept_targets"].to(device)
        coverage_token_ids = batch.get("coverage_token_ids")
        if coverage_token_ids is not None:
            coverage_token_ids = coverage_token_ids.to(device)
        with torch.set_grad_enabled(train):
            if attention_mask is not None:
                output = model(
                    images,
                    tokens,
                    attention_mask=attention_mask,
                    concept_targets=concept_targets,
                    concept_evidence_topk=cfg.decoder_concept_evidence_topk,
                    region_evidence_topk=cfg.decoder_region_evidence_topk,
                )
            else:
                output = model(images, tokens, concept_targets=concept_targets)
            loss, parts = compute_losses(
                output,
                tokens,
                concept_targets,
                model.pad_id,
                coverage_token_ids,
                cfg.lambda_concept,
                cfg.lambda_align,
                cfg.lambda_coverage,
                cfg.lambda_sparse,
                cfg.lambda_temp,
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
        bs = images.shape[0]
        n += bs
        for key, val in parts.items():
            totals[key] = totals.get(key, 0.0) + val * bs
        if use_batch_bar:
            iterator.set_postfix(_progress_postfix(totals, n))
    return {k: v / max(1, n) for k, v in totals.items()}


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)
    device = get_device(cfg.device)
    tqdm.write(f"Output directory: {out_dir}")
    tqdm.write(f"Device: {device}")
    tqdm.write(f"Dataset: {cfg.dataset} | decoder: {cfg.decoder_type} | LLM: {cfg.llm_name}")
    tqdm.write(f"Vision encoder: {cfg.vision_encoder_type} | name: {cfg.vision_encoder_name or 'default'} | frozen: {cfg.freeze_vision_encoder}")
    if _uses_hf_decoder(cfg):
        from transformers import AutoTokenizer

        tokenizer = _prepare_tokenizer(AutoTokenizer.from_pretrained(cfg.llm_name, **_tokenizer_kwargs(cfg)))
        tokenizer.save_pretrained(out_dir)
        tqdm.write("Loading training metadata and image paths...")
        train_records = load_split_records(cfg.data_root, "train", dataset=cfg.dataset, seed=cfg.seed)
        train_records = _limit_records(train_records, cfg.max_train_records, "training")
        tqdm.write(f"Loaded {len(train_records):,} training image-report records.")
        tqdm.write("Building concept vocabulary and graph priors...")
        if cfg.concept_source == "keywords":
            if cfg.relation_extractor == "none":
                concepts = build_concepts((r["keywords"] for r in train_records), max_concepts=cfg.max_concepts)
                concept_graph = {"concepts": concepts, "relations": [], "source": "keywords", "normalizer": "none"}
            else:
                concept_graph = build_concept_graph(
                    train_records,
                    cfg.max_concepts,
                    radgraph_path=None,
                    normalizer="rules",
                    relation_extractor=cfg.relation_extractor,
                    relation_extractor_model=cfg.relation_extractor_model,
                    relation_cache=out_dir / "relation_extraction_cache.json",
                    dataset=cfg.dataset,
                )
                concepts = concept_graph["concepts"]
        else:
            concept_graph = build_concept_graph(
                train_records,
                cfg.max_concepts,
                radgraph_path=cfg.radgraph_path,
                normalizer=cfg.concept_normalizer,
                normalizer_cache=out_dir / "concept_normalization_cache.json",
                llm_model=cfg.concept_normalizer_model,
                relation_extractor=cfg.relation_extractor,
                relation_extractor_model=cfg.relation_extractor_model,
                relation_cache=out_dir / "relation_extraction_cache.json",
                dataset=cfg.dataset,
            )
            concepts = concept_graph["concepts"]
        if not concepts:
            from .data import infer_concepts_from_reports
            concepts = infer_concepts_from_reports(train_records, max_concepts=cfg.max_concepts)
            concept_graph = {"concepts": concepts, "relations": [], "source": "fallback_report_terms", "normalizer": cfg.concept_normalizer}
        tqdm.write(f"Built {len(concepts):,} concepts and {len(concept_graph.get('relations', [])):,} graph relations.")
        save_json({"llm_name": cfg.llm_name, "decoder_type": cfg.decoder_type, "pad_token": tokenizer.pad_token, "source": "save_pretrained"}, out_dir / "tokenizer_meta.json")
        vocab = None
    else:
        vocab, concepts = build_artifacts(cfg.data_root, cfg.min_token_freq, cfg.max_vocab_size, cfg.max_concepts, dataset=cfg.dataset, seed=cfg.seed)
        train_records = load_split_records(cfg.data_root, "train", dataset=cfg.dataset, seed=cfg.seed)
        train_records = _limit_records(train_records, cfg.max_train_records, "training")
        if cfg.concept_source != "keywords":
            concept_graph = build_concept_graph(
                train_records,
                cfg.max_concepts,
                radgraph_path=cfg.radgraph_path,
                normalizer=cfg.concept_normalizer,
                normalizer_cache=out_dir / "concept_normalization_cache.json",
                llm_model=cfg.concept_normalizer_model,
                relation_extractor=cfg.relation_extractor,
                relation_extractor_model=cfg.relation_extractor_model,
                relation_cache=out_dir / "relation_extraction_cache.json",
                dataset=cfg.dataset,
            )
            concepts = concept_graph["concepts"] or concepts
        else:
            if cfg.relation_extractor == "none":
                concept_graph = {"concepts": concepts, "relations": [], "source": "keywords", "normalizer": "none"}
            else:
                concept_graph = build_concept_graph(
                    train_records,
                    cfg.max_concepts,
                    radgraph_path=None,
                    normalizer="rules",
                    relation_extractor=cfg.relation_extractor,
                    relation_extractor_model=cfg.relation_extractor_model,
                    relation_cache=out_dir / "relation_extraction_cache.json",
                    dataset=cfg.dataset,
                )
                concepts = concept_graph["concepts"] or concepts
        save_json(vocab.to_dict(), out_dir / "vocab.json")
    save_json({"concepts": concepts}, out_dir / "concepts.json")
    save_json(concept_graph, out_dir / "concept_graph.json")
    if cfg.use_report_memory:
        save_report_memory(train_records, out_dir / "report_memory.json", max_entries=cfg.report_memory_max_entries)
    cfg.save(out_dir / "config.json")

    if _uses_hf_decoder(cfg):
        valid_records = load_split_records(cfg.data_root, "valid", dataset=cfg.dataset, seed=cfg.seed)
        valid_records = _limit_records(valid_records, cfg.max_valid_records, "validation")
        train_ds = HFMedicalReportDataset(
            cfg.data_root,
            "train",
            tokenizer,
            concepts,
            cfg.dataset,
            cfg.image_size,
            cfg.max_report_len,
            cfg.seed,
            concept_graph.get("per_record_concepts", {}),
            records=train_records,
        )
        valid_ds = HFMedicalReportDataset(
            cfg.data_root,
            "valid",
            tokenizer,
            concepts,
            cfg.dataset,
            cfg.image_size,
            cfg.max_report_len,
            cfg.seed,
            records=valid_records,
        )
        pad_id, _, _ = _token_ids(tokenizer)
        collate = functools.partial(collate_hf_fn, pad_id=pad_id)
    else:
        valid_records = load_split_records(cfg.data_root, "valid", dataset=cfg.dataset, seed=cfg.seed)
        valid_records = _limit_records(valid_records, cfg.max_valid_records, "validation")
        train_ds = MedicalReportDataset(
            cfg.data_root,
            "train",
            vocab,
            concepts,
            cfg.dataset,
            cfg.image_size,
            cfg.max_report_len,
            cfg.seed,
            concept_graph.get("per_record_concepts", {}),
            records=train_records,
        )
        valid_ds = MedicalReportDataset(
            cfg.data_root,
            "valid",
            vocab,
            concepts,
            cfg.dataset,
            cfg.image_size,
            cfg.max_report_len,
            cfg.seed,
            records=valid_records,
        )
        collate = functools.partial(collate_fn, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate)
    tqdm.write(f"Train examples: {len(train_ds):,} | valid examples: {len(valid_ds):,}")
    tqdm.write(f"Train batches: {len(train_loader):,} | valid batches: {len(valid_loader):,}")
    if cfg.progress_style == "batch":
        tqdm.write("Live progress: batch bars show rolling epoch averages. Artifacts update after each epoch.")
    elif cfg.progress_style == "epoch":
        tqdm.write("Live progress: one epoch bar plus one summary per epoch. Artifacts update after each epoch.")
    else:
        tqdm.write("Live progress bars disabled. Artifacts update after each epoch.")

    if _uses_hf_decoder(cfg):
        model = _build_hf_model(cfg, tokenizer, concepts, concept_graph).to(device)
    else:
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
            _anatomy_concept_prior_from_graph(cfg, concepts, concept_graph),
            cfg.relation_prior_weight,
            cfg.use_anatomy,
            cfg.vision_encoder_type,
            cfg.vision_encoder_name,
            cfg.vision_checkpoint,
            cfg.freeze_vision_encoder,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    history = []
    epoch_iter = range(1, cfg.epochs + 1)
    epoch_bar = tqdm(epoch_iter, desc="epochs", dynamic_ncols=True) if cfg.progress_style != "none" else epoch_iter
    for epoch in epoch_bar:
        train_metrics = run_epoch(model, train_loader, optimizer, cfg, device, train=True, epoch=epoch)
        valid_metrics = run_epoch(model, valid_loader, optimizer, cfg, device, train=False, epoch=epoch)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        save_json(history, out_dir / "history.json")
        _write_history_csv(history, out_dir)
        _plot_history(history, out_dir)
        summary = (
            f"epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_metrics.get('loss', float('nan')):.4f} "
            f"valid_loss={valid_metrics.get('loss', float('nan')):.4f} "
            f"valid_rep={valid_metrics.get('rep_loss', float('nan')):.4f}"
        )
        tqdm.write(summary)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out_dir / "best_model.pt")
            tqdm.write(f"New best validation loss: {best:.4f}; checkpoint saved.")
        if cfg.progress_style != "none":
            epoch_bar.set_postfix(best=f"{best:.4f}", valid=f"{valid_metrics['loss']:.4f}")
    print(f"Best validation loss: {best:.4f}")
    print(f"Saved checkpoint to {Path(out_dir) / 'best_model.pt'}")
    print(f"Saved training plot to {Path(out_dir) / 'training_progress.png'}")
    print(f"Saved training CSV to {Path(out_dir) / 'training_progress.csv'}")


if __name__ == "__main__":
    main()
