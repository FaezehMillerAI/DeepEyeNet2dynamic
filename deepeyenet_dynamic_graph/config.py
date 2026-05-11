from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import json


@dataclass
class Config:
    data_root: str
    dataset: str = "deepeyenet"
    output_dir: str = "outputs/deepeyenet_dynamic_graph"
    image_size: int = 224
    patch_grid: int = 4
    max_report_len: int = 96
    min_token_freq: int = 1
    max_vocab_size: int = 12000
    max_concepts: int = 128
    embed_dim: int = 256
    hidden_dim: int = 256
    graph_steps: int = 1
    use_anatomy: bool = True
    disable_counterfactuals: bool = False
    max_interactive_examples: int = 12
    dropout: float = 0.2
    batch_size: int = 8
    epochs: int = 15
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 0
    seed: int = 42
    lambda_concept: float = 0.4
    lambda_align: float = 0.1
    lambda_sparse: float = 0.01
    lambda_temp: float = 0.05
    grad_clip: float = 1.0
    topk_evidence: int = 3
    device: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls(**json.loads(Path(path).read_text()))
