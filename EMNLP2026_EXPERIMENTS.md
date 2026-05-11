# EMNLP 2026 Experiment Plan

This project should be positioned as a faithful multimodal explanation framework for medical report generation, not only as a report-generation model.

## Core Claim

The model generates reports through a dynamic explanation graph:

```text
image patch -> anatomy node -> finding/concept node -> generated report text
```

Explanations are evaluated through counterfactual interventions:

```text
remove patch/anatomy/finding evidence -> measure whether the finding weakens
```

## Required Main Experiments

Run each experiment on DeepEyeNet and IU-XRay with at least three random seeds.

1. Full anatomy-aware dynamic explanation graph.
2. Full graph with biomedical LLM decoder (`--decoder-type llm --llm-name microsoft/BioGPT`).
3. Frozen LLM prefix tuning (`--freeze-llm`).
4. GRU decoder baseline (`--decoder-type gru`).
5. No anatomy layer (`--no-anatomy`).
6. No graph sparsity (`--lambda-sparse 0`).
7. No temporal graph consistency (`--lambda-temp 0`).
8. No sparsity and no temporal consistency.
9. Patch-grid sensitivity (`--patch-grid 3`, `4`, `5`).

## Metrics

Report generation:

- BLEU-1/2/3/4
- ROUGE-L
- METEOR
- BERTScore where feasible

Clinical fidelity:

- Concept micro/macro precision, recall, F1
- Exact concept match
- IU-XRay: add CheXbert/CheXpert label F1 if available

Explanation quality:

- Region-concept entropy
- Token-concept entropy
- Temporal graph drift
- Top-k concept hit rate
- Patch counterfactual confidence drop
- Anatomy counterfactual confidence drop
- Finding-node counterfactual confidence drop
- Positive-drop rate for each intervention

Qualitative analysis:

- Evidence heatmaps
- Anatomy-aware graph diagrams
- Interactive hover explanation cases
- Failure cases where counterfactual drops are negative or near zero

## Reviewer-Sensitive Framing

Do not claim that attention alone is explanation. The paper should explicitly state:

> We use graph attention as a proposed evidence path and evaluate its faithfulness using counterfactual graph and image interventions.

This distinction is crucial for EMNLP reviewers.

## Suggested Title

Dynamic Anatomy-Aware Explanation Graphs for Faithful Medical Report Generation
