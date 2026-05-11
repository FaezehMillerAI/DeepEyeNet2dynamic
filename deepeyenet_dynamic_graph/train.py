from __future__ import annotations

import argparse
import functools
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import MedicalReportDataset, build_artifacts, collate_fn
from .model import DynamicGraphCaptioner, compute_losses
from .utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset", choices=["deepeyenet", "iuxray"], default="deepeyenet")
    parser.add_argument("--output-dir", default="outputs/deepeyenet_dynamic_graph")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-report-len", type=int, default=96)
    parser.add_argument("--max-concepts", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
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
        max_report_len=args.max_report_len,
        max_concepts=args.max_concepts,
        num_workers=args.num_workers,
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
        concept_targets = batch["concept_targets"].to(device)
        with torch.set_grad_enabled(train):
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
    vocab, concepts = build_artifacts(cfg.data_root, cfg.min_token_freq, cfg.max_vocab_size, cfg.max_concepts, dataset=cfg.dataset, seed=cfg.seed)
    save_json(vocab.to_dict(), out_dir / "vocab.json")
    save_json({"concepts": concepts}, out_dir / "concepts.json")
    cfg.save(out_dir / "config.json")

    train_ds = MedicalReportDataset(cfg.data_root, "train", vocab, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
    valid_ds = MedicalReportDataset(cfg.data_root, "valid", vocab, concepts, cfg.dataset, cfg.image_size, cfg.max_report_len, cfg.seed)
    collate = functools.partial(collate_fn, pad_id=vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=collate)

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
