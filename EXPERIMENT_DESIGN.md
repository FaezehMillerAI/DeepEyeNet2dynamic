# Experiment Design and Publication Notes

## Methodological Adaptation

The markdown proposal defines a dynamic explanation graph with region, concept, and token nodes. DeepEyeNet is retinal rather than chest X-ray data, so this implementation maps:

- **Region nodes** to retinal image patches from a fixed spatial grid.
- **Concept nodes** to normalized DeepEyeNet `Keywords`.
- **Token nodes** to generated `report_text` tokens.

This preserves the proposed dynamic graph mechanism while avoiding unsupported assumptions about chest anatomy.

## Design Choices

**Fixed patch regions instead of a segmentation network.** DeepEyeNet metadata does not provide lesion masks. A fixed grid is transparent, deterministic, and suitable for faithfulness probes. If lesion annotations become available, the `RegionEncoder` can be replaced without changing the decoder/evaluator.

**Compact CNN region encoder.** The dataset is small, so a lightweight encoder is less likely to overfit than a large ViT trained from scratch. The code is modular enough to replace it with a pretrained ViT or ophthalmology encoder in a future ablation.

**Keyword-derived concept vocabulary.** The dataset already supplies clinical keywords. Using these as concept nodes gives weak supervision for graph alignment and enables clinically interpretable F1 scores.

**GRU decoder with dynamic graph context.** A GRU is efficient on Colab and easier to audit. It conditions every token on region-concept and concept-token evidence, implementing the core dynamic graph idea without needing a large language model.

**Post-training faithfulness and counterfactual probes.** Evidence masking is evaluated after training to keep the training loop stable and affordable. The metric asks whether suppressing the highest-scoring graph region reduces the predicted concept confidence.

## Evaluation Suite

Language generation:

- BLEU-1/2/3/4
- ROUGE-L
- METEOR

Clinical fidelity:

- Micro precision, recall, F1
- Macro precision, recall, F1
- Exact concept match

Explanation graph behavior:

- Region-concept entropy
- Token-concept entropy
- Temporal graph drift
- Top-k concept hit rate against keyword labels

Faithfulness and counterfactual sensitivity:

- Mean and median confidence drop after masking top-evidence region
- Distribution plot of confidence drops

Publication visualizations:

- Metric summary bar chart
- Top concept TP/FP/FN heatmap
- Evidence heatmaps over retinal images
- Dynamic graph diagrams linking region nodes to concept nodes
- Counterfactual evidence-drop distribution

## Recommended Ablations

Run these for a journal paper:

1. Full dynamic graph model.
2. Remove temporal consistency loss.
3. Remove sparsity loss.
4. Remove concept supervision.
5. Static graph variant by freezing region-concept edges after initialization.
6. Different patch grids: 3x3, 4x4, 5x5.

Report mean and standard deviation across at least three random seeds because DeepEyeNet is small.
