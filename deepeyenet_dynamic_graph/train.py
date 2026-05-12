from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import HFMedicalReportDataset, MedicalReportDataset, anatomy_prior_matrix, build_artifacts, collate_fn, collate_hf_fn, get_anatomy_names, load_split_records
from .model import DynamicGraphCaptioner, GraphPrefixLLMCaptioner, compute_losses
from .utils import ensure_dir, get_device, save_json, set_seed
from .vocab import build_concepts


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["deepeyenet", "iuxray"], default="deepeyenet")
    parser.add_argument("--output-dir", default="outputs/deepeyenet_dynamic_graph")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-grid", type=int, default=4)
    parser.add_argument("--max-report-len", type=int, default=96)
    parser.add_argument("--max-concepts", type=int, default=128)
    parser.add_argument("--decoder-type", choices=["llm", "gru"], default="llm")
    parser.add_argument("--llm-name", default="distilgpt2")
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--prefix-length", type=int, default=4)
    parser.add_argument("--graph-steps", type=int, default=1)
    parser.add_argument("--lambda-concept", type=float, default=0.4)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--lambda-sparse", type=float, default=0.01)
    parser.add_argument("--lambda-temp", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--no-anatomy", action="store_true")
    parser.add_argument("--disable-counterfactuals", action="store_true")
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
        decoder_type=args.decoder_type,
        llm_name=args.llm_name,
        freeze_llm=args.freeze_llm,
        prefix_length=args.prefix_length,
        graph_steps=args.graph_steps,
        lambda_concept=args.lambda_concept,
        lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse,
        lambda_temp=args.lambda_temp,
        num_workers=args.num_workers,
        grad_clip=args.grad_clip,
        use_anatomy=not args.no_anatomy,
        disable_counterfactuals=args.disable_counterfactuals,
        device=args.device,
    )
    return cfg


def run_epoch(model, loader, optimizer, cfg: Config, device: torch.device, train: bool) -> dict[str, float]:
    model.train(train)
    totals: dict[str, float] = {}
    n = 0
    iterator = tqdm(loader, desc="train" if train else "valid", leave=False)
    for batch in iterator:
        images = batch["image"].to(device)
        tokens = batch["tokens"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        concept_targets = batch["concept_targets"].to(device)
        with torch.set_grad_enabled(train):
            if attention_mask is not None:
                output = model(images, tokens, attention_mask=attention_mask)
            else:
                output = model(images, tokens)
            loss, parts = compute_losses(
                output,
                tokens,
                concept_targets,
                model.pad_id,
                cfg.lambda_concept,
                cfg.lambda_align,
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
        iterator.set_postfix(loss=parts["loss"])
    return {k: v / max(1, n) for k, v in totals.items()}


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    out_dir = ensure_dir(cfg.output_dir)
    device = get_device(cfg.device)
    if cfg.decoder_type == "llm":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(out_dir)
        train_records = load_split_records(cfg.data_root, "train", dataset=cfg.dataset, seed=cfg.seed)
        concepts = build_concepts((r["keywords"] for r in train_records), max_concepts=cfg.max_concepts)
        if not concepts:
            from .data import infer_concepts_from_reports

            concepts = infer_concepts_from_reports(train_records, max_concepts=cfg.max_concepts)
        save_json({"llm_name": cfg.llm_name, "pad_token": tokenizer.pad_token, "source": "save_pretrained"}, out_dir / "tokenizer_meta.json")
        vocab = None
    else:
        vocab, concepts = build_artifacts(cfg.data_root, cfg.min_token_freq, cfg.max_vocab_size, cfg.max_concepts, dataset=cfg.dataset, seed=cfg.seed)
        save_json(vocab.to_dict(), out_dir / "vocab.json")
    save_json({"concepts": concepts}, out_dir / "concepts.json")
    cfg.save(out_dir / "config.json")

    if cfg.decoder_type == "llm":
        train_ds = HFMedicalReportDataset(cfg.data_root, "train", tokenizer, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
        valid_ds = HFMedicalReportDataset(cfg.data_root, "valid", tokenizer, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
        collate = functools.partial(collate_hf_fn, pad_id=tokenizer.pad_token_id)
    else:
        train_ds = MedicalReportDataset(cfg.data_root, "train", vocab, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
        valid_ds = MedicalReportDataset(cfg.data_root, "valid", vocab, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
        collate = functools.partial(collate_fn, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate)

    if cfg.decoder_type == "llm":
        bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
        model = GraphPrefixLLMCaptioner(
            cfg.llm_name,
            concepts,
            tokenizer.pad_token_id,
            bos_id,
            tokenizer.eos_token_id,
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
        ).to(device)
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
            cfg.use_anatomy,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = float("inf")
    history = []
    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, cfg, device, train=True)
        valid_metrics = run_epoch(model, valid_loader, optimizer, cfg, device, train=False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        save_json(history, out_dir / "history.json")
        print(row)
        if valid_metrics["loss"] < best:
            best = valid_metrics["loss"]
            torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out_dir / "best_model.pt")
    print(f"Best validation loss: {best:.4f}")
    print(f"Saved checkpoint to {Path(out_dir) / 'best_model.pt'}")


if __name__ == "__main__":
    main()
