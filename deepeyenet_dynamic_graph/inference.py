from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .config import Config
from .data import anatomy_prior_matrix, get_anatomy_names, make_transforms
from .model import DynamicGraphCaptioner, GraphPrefixLLMCaptioner, GraphSeq2SeqCaptioner
from .utils import get_device, load_json
from .vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--max-report-len", type=int, default=96)
    parser.add_argument("--device", default="auto")
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
    if _uses_hf_decoder(cfg):
        from transformers import AutoTokenizer

        tokenizer_source = run_dir if (run_dir / "tokenizer_config.json").exists() else cfg.llm_name
        tokenizer = _prepare_tokenizer(AutoTokenizer.from_pretrained(tokenizer_source))
        model = _build_hf_model(cfg, tokenizer, concepts).to(device)
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
