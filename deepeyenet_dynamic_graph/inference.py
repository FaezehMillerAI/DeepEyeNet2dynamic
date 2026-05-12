from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import Config
from .data import anatomy_prior_matrix, get_anatomy_names, make_transforms
from .model import DynamicGraphCaptioner, GraphPrefixLLMCaptioner
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
    if not (run_dir / "tokenizer_config.json").exists() and not (run_dir / "tokenizer.json").exists() and (run_dir / "vocab.json").exists():
        cfg.decoder_type = "gru"
    cfg.device = args.device
    concepts = load_json(run_dir / "concepts.json")["concepts"]
    device = get_device(cfg.device)
    if cfg.decoder_type == "llm":
        from transformers import AutoTokenizer

        tokenizer_source = run_dir if (run_dir / "tokenizer_config.json").exists() else cfg.llm_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
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
        decoder = tokenizer
    else:
        vocab = Vocabulary.from_dict(load_json(run_dir / "vocab.json"))
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
        decoder = vocab
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    transform = make_transforms(cfg.image_size, train=False)
    image = transform(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
    output, gen_tokens = model.generate(image, max_len=args.max_report_len)
    if hasattr(decoder, "itos"):
        report = decoder.decode(gen_tokens[0].cpu().tolist())
    else:
        report = decoder.decode(gen_tokens[0].cpu().tolist(), skip_special_tokens=True)
    concept_probs = torch.sigmoid(output.concept_logits[0]).cpu()
    top = torch.topk(concept_probs, k=min(8, len(concepts)))
    print("Generated report:")
    print(report)
    print("\nTop concepts:")
    for score, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"{concepts[idx]}: {score:.3f}")


if __name__ == "__main__":
    main()
