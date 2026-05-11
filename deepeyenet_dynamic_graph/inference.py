from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import Config
from .data import anatomy_prior_matrix, get_anatomy_names, make_transforms
from .model import DynamicGraphCaptioner
from .utils import get_device, load_json
from .vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-report-len", type=int, default=96)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    run_dir = Path(args.checkpoint).parent
    cfg = Config.load(run_dir / "config.json")
    cfg.device = args.device
    vocab = Vocabulary.from_dict(load_json(run_dir / "vocab.json"))
    concepts = load_json(run_dir / "concepts.json")["concepts"]
    device = get_device(cfg.device)
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
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    transform = make_transforms(cfg.image_size, train=False)
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    output, gen_tokens = model.generate(image, max_len=args.max_report_len)
    report = vocab.decode(gen_tokens[0].cpu().tolist())
    concept_probs = torch.sigmoid(output.concept_logits[0]).cpu()
    top = torch.topk(concept_probs, k=min(8, len(concepts)))
    print("Generated report:")
    print(report)
    print("\nTop concepts:")
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{concepts[idx]}: {score:.3f}")


if __name__ == "__main__":
    main()
