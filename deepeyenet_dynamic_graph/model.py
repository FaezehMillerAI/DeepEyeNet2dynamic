from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecodeOutput:
    logits: torch.Tensor
    concept_logits: torch.Tensor
    rc_edges: torch.Tensor
    token_concept_edges: torch.Tensor
    region_features: torch.Tensor
    concept_features: torch.Tensor
    region_anatomy_edges: torch.Tensor | None = None
    anatomy_concept_edges: torch.Tensor | None = None
    anatomy_features: torch.Tensor | None = None
    lm_loss: torch.Tensor | None = None


class RegionEncoder(nn.Module):
    """Compact patch encoder for retinal region nodes."""

    def __init__(self, embed_dim: int, patch_grid: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.patch_grid = patch_grid
        self.num_regions = patch_grid * patch_grid
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((patch_grid, patch_grid)),
        )
        self.proj = nn.Sequential(nn.Flatten(2), nn.Dropout(dropout))
        self.linear = nn.Linear(64, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(self.num_regions, embed_dim) * 0.02)
        self.anatomy_embed = nn.Embedding(4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.cnn(x)
        feat = self.proj(feat).transpose(1, 2)
        feat = self.linear(feat)
        anatomy_ids = self._anatomy_ids(x.device)
        return feat + self.pos_embed.unsqueeze(0) + self.anatomy_embed(anatomy_ids).unsqueeze(0)

    def _anatomy_ids(self, device: torch.device) -> torch.Tensor:
        ids = []
        for y in range(self.patch_grid):
            for x in range(self.patch_grid):
                ids.append((y >= self.patch_grid / 2) * 2 + (x >= self.patch_grid / 2))
        return torch.tensor(ids, device=device, dtype=torch.long)


class DynamicGraphCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        concept_names: list[str],
        pad_id: int,
        bos_id: int,
        eos_id: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        patch_grid: int = 4,
        dropout: float = 0.2,
        graph_steps: int = 1,
        anatomy_names: list[str] | None = None,
        region_anatomy_prior: torch.Tensor | None = None,
        use_anatomy: bool = True,
    ) -> None:
        super().__init__()
        self.concept_names = concept_names
        self.num_concepts = len(concept_names)
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.graph_steps = graph_steps
        self.anatomy_names = anatomy_names or []
        self.num_anatomy = len(self.anatomy_names)
        self.use_anatomy = use_anatomy and self.num_anatomy > 0

        self.region_encoder = RegionEncoder(embed_dim, patch_grid=patch_grid, dropout=dropout)
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.concept_embed = nn.Parameter(torch.randn(self.num_concepts, embed_dim) * 0.02)
        self.anatomy_embed = nn.Parameter(torch.randn(max(1, self.num_anatomy), embed_dim) * 0.02)
        self.region_proj = nn.Linear(embed_dim, embed_dim)
        self.anatomy_proj = nn.Linear(embed_dim, embed_dim)
        self.concept_proj = nn.Linear(embed_dim, embed_dim)
        self.graph_msg = nn.Linear(embed_dim, embed_dim)
        self.init_hidden = nn.Linear(embed_dim, hidden_dim)
        self.decoder = nn.GRUCell(embed_dim + embed_dim + embed_dim, hidden_dim)
        self.hidden_to_graph = nn.Linear(hidden_dim, embed_dim)
        self.out = nn.Linear(hidden_dim, vocab_size)
        self.concept_head = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        if region_anatomy_prior is None:
            region_anatomy_prior = torch.full((patch_grid * patch_grid, max(1, self.num_anatomy)), 1.0 / max(1, self.num_anatomy))
        self.register_buffer("region_anatomy_prior", region_anatomy_prior.float())

    def compute_region_concept_edges(
        self, region_features: torch.Tensor, concept_features: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        r = self.region_proj(region_features)
        c = self.concept_proj(concept_features)
        if hidden is not None:
            c = c + self.hidden_to_graph(hidden).unsqueeze(1)
        scores = torch.matmul(r, c.transpose(1, 2)) / (r.shape[-1] ** 0.5)
        return F.softmax(scores, dim=-1)

    def compute_region_anatomy_edges(self, region_features: torch.Tensor) -> torch.Tensor:
        if not self.use_anatomy:
            batch = region_features.shape[0]
            return torch.zeros(batch, region_features.shape[1], max(1, self.num_anatomy), device=region_features.device)
        anatomy_features = self.anatomy_embed[: self.num_anatomy].unsqueeze(0).expand(region_features.shape[0], -1, -1)
        scores = torch.matmul(self.region_proj(region_features), self.anatomy_proj(anatomy_features).transpose(1, 2))
        scores = scores / (region_features.shape[-1] ** 0.5)
        prior = self.region_anatomy_prior[:, : self.num_anatomy].to(region_features.device).clamp_min(1e-8).log().unsqueeze(0)
        return F.softmax(scores + prior, dim=-1)

    def compute_anatomy_concept_edges(self, anatomy_features: torch.Tensor, concept_features: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(self.anatomy_proj(anatomy_features), self.concept_proj(concept_features).transpose(1, 2))
        scores = scores / (anatomy_features.shape[-1] ** 0.5)
        return F.softmax(scores, dim=-1)

    def build_graph_features(
        self,
        region_features: torch.Tensor,
        concept_features: torch.Tensor,
        suppress_anatomy_ids: torch.Tensor | None = None,
        suppress_concept_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if self.use_anatomy:
            ra_edges = self.compute_region_anatomy_edges(region_features)
            anatomy_features = self.anatomy_embed[: self.num_anatomy].unsqueeze(0).expand(region_features.shape[0], -1, -1)
            anatomy_msg = torch.matmul(ra_edges.transpose(1, 2), region_features) / max(1, region_features.shape[1])
            anatomy_features = F.gelu(anatomy_features + self.graph_msg(anatomy_msg))
            if suppress_anatomy_ids is not None:
                anatomy_features = anatomy_features.clone()
                for b, anatomy_id in enumerate(suppress_anatomy_ids.tolist()):
                    anatomy_features[b, int(anatomy_id)] = 0.0
            ac_edges = self.compute_anatomy_concept_edges(anatomy_features, concept_features)
            concept_msg = torch.matmul(ac_edges.transpose(1, 2), anatomy_features) / max(1, self.num_anatomy)
            rc_edges = torch.matmul(ra_edges, ac_edges)
        else:
            ra_edges = None
            ac_edges = None
            anatomy_features = None
            rc_edges = self.compute_region_concept_edges(region_features, concept_features)
            concept_msg = torch.matmul(rc_edges.transpose(1, 2), region_features) / max(1, region_features.shape[1])
        for _ in range(self.graph_steps):
            concept_features = F.gelu(concept_features + self.graph_msg(concept_msg))
        if suppress_concept_ids is not None:
            concept_features = concept_features.clone()
            for b, concept_id in enumerate(suppress_concept_ids.tolist()):
                concept_features[b, int(concept_id)] = 0.0
        return concept_features, ra_edges, ac_edges, anatomy_features, rc_edges

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        suppress_anatomy_ids: torch.Tensor | None = None,
        suppress_concept_ids: torch.Tensor | None = None,
    ) -> DecodeOutput:
        batch, seq_len = tokens.shape
        region_features = self.region_encoder(images)
        concept_features = self.concept_embed.unsqueeze(0).expand(batch, -1, -1)
        concept_features, ra_edges, ac_edges, anatomy_features, rc0 = self.build_graph_features(region_features, concept_features, suppress_anatomy_ids, suppress_concept_ids)
        pooled = region_features.mean(dim=1)
        hidden = torch.tanh(self.init_hidden(pooled))

        logits = []
        rc_edges = []
        token_concept_edges = []
        for t in range(seq_len - 1):
            rc_t = self.compute_region_concept_edges(region_features, concept_features, hidden)
            graph_context = torch.matmul(rc_t.mean(dim=1).unsqueeze(1), concept_features).squeeze(1)
            token_emb = self.token_embed(tokens[:, t])
            concept_scores = torch.matmul(concept_features, self.hidden_to_graph(hidden).unsqueeze(-1)).squeeze(-1)
            cy_t = F.softmax(concept_scores, dim=-1)
            concept_context = torch.matmul(cy_t.unsqueeze(1), concept_features).squeeze(1)
            hidden = self.decoder(torch.cat([token_emb, graph_context, concept_context], dim=-1), hidden)
            hidden = self.dropout(hidden)
            logits.append(self.out(hidden))
            rc_edges.append(rc_t)
            token_concept_edges.append(cy_t)

        logits_t = torch.stack(logits, dim=1)
        rc_edges_t = torch.stack(rc_edges, dim=1)
        token_concept_t = torch.stack(token_concept_edges, dim=1)
        concept_logits = self.concept_head(concept_features).squeeze(-1)
        return DecodeOutput(logits_t, concept_logits, rc_edges_t, token_concept_t, region_features, concept_features, ra_edges, ac_edges, anatomy_features)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int = 96) -> DecodeOutput:
        batch = images.shape[0]
        tokens = torch.full((batch, max_len), self.pad_id, dtype=torch.long, device=images.device)
        tokens[:, 0] = self.bos_id
        generated = []

        region_features = self.region_encoder(images)
        concept_features = self.concept_embed.unsqueeze(0).expand(batch, -1, -1)
        concept_features, ra_edges, ac_edges, anatomy_features, rc0 = self.build_graph_features(region_features, concept_features)
        hidden = torch.tanh(self.init_hidden(region_features.mean(dim=1)))

        logits = []
        rc_edges = []
        token_concept_edges = []
        prev = tokens[:, 0]
        for t in range(max_len - 1):
            rc_t = self.compute_region_concept_edges(region_features, concept_features, hidden)
            graph_context = torch.matmul(rc_t.mean(dim=1).unsqueeze(1), concept_features).squeeze(1)
            concept_scores = torch.matmul(concept_features, self.hidden_to_graph(hidden).unsqueeze(-1)).squeeze(-1)
            cy_t = F.softmax(concept_scores, dim=-1)
            concept_context = torch.matmul(cy_t.unsqueeze(1), concept_features).squeeze(1)
            hidden = self.decoder(torch.cat([self.token_embed(prev), graph_context, concept_context], dim=-1), hidden)
            step_logits = self.out(hidden)
            prev = step_logits.argmax(dim=-1)
            generated.append(prev)
            logits.append(step_logits)
            rc_edges.append(rc_t)
            token_concept_edges.append(cy_t)
        gen_tokens = torch.stack(generated, dim=1)
        concept_logits = self.concept_head(concept_features).squeeze(-1)
        return DecodeOutput(
            torch.stack(logits, dim=1),
            concept_logits,
            torch.stack(rc_edges, dim=1),
            torch.stack(token_concept_edges, dim=1),
            region_features,
            concept_features,
            ra_edges,
            ac_edges,
            anatomy_features,
        ), gen_tokens


def compute_losses(
    output: DecodeOutput,
    target_tokens: torch.Tensor,
    concept_targets: torch.Tensor,
    pad_id: int,
    lambda_concept: float = 0.4,
    lambda_align: float = 0.1,
    lambda_sparse: float = 0.01,
    lambda_temp: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    if output.lm_loss is not None:
        rep_loss = output.lm_loss
    else:
        lm_targets = target_tokens[:, 1 : output.logits.shape[1] + 1]
        rep_loss = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), lm_targets.reshape(-1), ignore_index=pad_id)
    concept_loss = F.binary_cross_entropy_with_logits(output.concept_logits, concept_targets)
    concept_probs = torch.sigmoid(output.concept_logits)
    align_loss = F.binary_cross_entropy(concept_probs.clamp(1e-4, 1 - 1e-4), concept_targets)
    edge_probs = output.token_concept_edges.clamp_min(1e-8)
    sparse_loss = -(edge_probs * edge_probs.log()).sum(dim=-1).mean()
    temp_loss = torch.tensor(0.0, device=target_tokens.device)
    if output.rc_edges.shape[1] > 1:
        temp_loss = (output.rc_edges[:, 1:] - output.rc_edges[:, :-1]).abs().mean()
    total = rep_loss + lambda_concept * concept_loss + lambda_align * align_loss + lambda_sparse * sparse_loss + lambda_temp * temp_loss
    return total, {
        "loss": float(total.detach().cpu()),
        "rep_loss": float(rep_loss.detach().cpu()),
        "concept_loss": float(concept_loss.detach().cpu()),
        "align_loss": float(align_loss.detach().cpu()),
        "sparse_loss": float(sparse_loss.detach().cpu()),
        "temp_loss": float(temp_loss.detach().cpu()),
    }


class GraphPrefixLLMCaptioner(DynamicGraphCaptioner):
    """Anatomy-aware graph encoder coupled to a causal LLM via soft prefixes."""

    def __init__(
        self,
        llm_name: str,
        concept_names: list[str],
        pad_id: int,
        bos_id: int,
        eos_id: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        patch_grid: int = 4,
        dropout: float = 0.2,
        graph_steps: int = 1,
        anatomy_names: list[str] | None = None,
        region_anatomy_prior: torch.Tensor | None = None,
        use_anatomy: bool = True,
        freeze_llm: bool = False,
        prefix_length: int = 4,
    ) -> None:
        from transformers import AutoModelForCausalLM

        dummy_vocab_size = max(int(pad_id or 0), int(bos_id or 0), int(eos_id or 0), 8) + 1
        super().__init__(
            vocab_size=dummy_vocab_size,
            concept_names=concept_names,
            pad_id=pad_id,
            bos_id=bos_id,
            eos_id=eos_id,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            patch_grid=patch_grid,
            dropout=dropout,
            graph_steps=graph_steps,
            anatomy_names=anatomy_names,
            region_anatomy_prior=region_anatomy_prior,
            use_anatomy=use_anatomy,
        )
        self.llm_name = llm_name
        self.llm = AutoModelForCausalLM.from_pretrained(llm_name)
        self.llm_dim = self.llm.config.hidden_size
        self.prefix_length = prefix_length
        self.prefix_offset = nn.Parameter(torch.randn(prefix_length, self.llm_dim) * 0.02)
        self.prefix_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.llm_dim),
        )
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

    def _graph_prefix(
        self,
        region_features: torch.Tensor,
        concept_features: torch.Tensor,
        anatomy_features: torch.Tensor | None,
        rc_edges: torch.Tensor,
    ) -> torch.Tensor:
        batch = region_features.shape[0]
        region_summary = region_features.mean(dim=1)
        concept_scores = torch.sigmoid(self.concept_head(concept_features).squeeze(-1))
        concept_summary = torch.matmul(concept_scores.unsqueeze(1), concept_features).squeeze(1)
        if anatomy_features is None:
            anatomy_summary = region_summary
        else:
            anatomy_summary = anatomy_features.mean(dim=1)
        base = torch.cat([region_summary, anatomy_summary, concept_summary], dim=-1)
        prefix = self.prefix_proj(base).unsqueeze(1).expand(batch, self.prefix_length, self.llm_dim)
        return prefix + self.prefix_offset.unsqueeze(0)

    def _graph_forward(
        self,
        images: torch.Tensor,
        suppress_anatomy_ids: torch.Tensor | None = None,
        suppress_concept_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        region_features = self.region_encoder(images)
        concept_features = self.concept_embed.unsqueeze(0).expand(images.shape[0], -1, -1)
        concept_features, ra_edges, ac_edges, anatomy_features, rc_edges = self.build_graph_features(
            region_features,
            concept_features,
            suppress_anatomy_ids,
            suppress_concept_ids,
        )
        prefix = self._graph_prefix(region_features, concept_features, anatomy_features, rc_edges)
        return prefix, region_features, ra_edges, ac_edges, anatomy_features, concept_features

    def forward(
        self,
        images: torch.Tensor,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        suppress_anatomy_ids: torch.Tensor | None = None,
        suppress_concept_ids: torch.Tensor | None = None,
    ) -> DecodeOutput:
        prefix, region_features, ra_edges, ac_edges, anatomy_features, concept_features = self._graph_forward(
            images, suppress_anatomy_ids, suppress_concept_ids
        )
        text_embeds = self.llm.get_input_embeddings()(tokens)
        inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
        if attention_mask is None:
            attention_mask = (tokens != self.pad_id).long()
        prefix_mask = torch.ones(tokens.shape[0], self.prefix_length, dtype=attention_mask.dtype, device=tokens.device)
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        labels = tokens.clone()
        labels[attention_mask == 0] = -100
        prefix_labels = torch.full((tokens.shape[0], self.prefix_length), -100, dtype=torch.long, device=tokens.device)
        labels = torch.cat([prefix_labels, labels], dim=1)
        lm_out = self.llm(inputs_embeds=inputs_embeds, attention_mask=full_mask, labels=labels)
        concept_logits = self.concept_head(concept_features).squeeze(-1)
        rc_edges = torch.matmul(ra_edges, ac_edges) if ra_edges is not None and ac_edges is not None else self.compute_region_concept_edges(region_features, concept_features)
        steps = max(1, tokens.shape[1] - 1)
        rc_seq = rc_edges.unsqueeze(1).expand(-1, steps, -1, -1)
        concept_probs = torch.softmax(concept_logits, dim=-1)
        token_concept = concept_probs.unsqueeze(1).expand(-1, steps, -1)
        logits = lm_out.logits[:, self.prefix_length :, :]
        return DecodeOutput(logits, concept_logits, rc_seq, token_concept, region_features, concept_features, ra_edges, ac_edges, anatomy_features, lm_out.loss)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, max_len: int = 96) -> tuple[DecodeOutput, torch.Tensor]:
        prefix, region_features, ra_edges, ac_edges, anatomy_features, concept_features = self._graph_forward(images)
        batch = images.shape[0]
        start_id = self.bos_id if self.bos_id is not None and self.bos_id >= 0 else self.eos_id
        generated = torch.full((batch, 1), start_id, dtype=torch.long, device=images.device)
        logits_steps = []
        finished = torch.zeros(batch, dtype=torch.bool, device=images.device)
        for _ in range(max_len - 1):
            text_embeds = self.llm.get_input_embeddings()(generated)
            inputs_embeds = torch.cat([prefix, text_embeds], dim=1)
            mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=images.device)
            out = self.llm(inputs_embeds=inputs_embeds, attention_mask=mask)
            next_logits = out.logits[:, -1, :]
            next_id = next_logits.argmax(dim=-1)
            next_id = torch.where(finished, torch.full_like(next_id, self.pad_id), next_id)
            generated = torch.cat([generated, next_id[:, None]], dim=1)
            logits_steps.append(next_logits)
            if self.eos_id is not None:
                finished |= next_id == self.eos_id
            if bool(finished.all()):
                break
        gen_tokens = generated[:, 1:]
        concept_logits = self.concept_head(concept_features).squeeze(-1)
        rc_edges = torch.matmul(ra_edges, ac_edges) if ra_edges is not None and ac_edges is not None else self.compute_region_concept_edges(region_features, concept_features)
        steps = max(1, gen_tokens.shape[1])
        rc_seq = rc_edges.unsqueeze(1).expand(-1, steps, -1, -1)
        token_concept = torch.softmax(concept_logits, dim=-1).unsqueeze(1).expand(-1, steps, -1)
        logits = torch.stack(logits_steps, dim=1) if logits_steps else torch.empty(batch, 0, self.llm.config.vocab_size, device=images.device)
        return DecodeOutput(logits, concept_logits, rc_seq, token_concept, region_features, concept_features, ra_edges, ac_edges, anatomy_features), gen_tokens
