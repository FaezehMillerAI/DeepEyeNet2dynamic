# DeepEyeNet Dynamic Explanation Graph

This repository implements the methodology in `formal_journal_methodology_dynamic_graph.md` as a modular, Google Colab friendly PyTorch project for the DeepEyeNet retinal image-report dataset.

The original markdown describes a dynamic explanation graph for chest X-ray report generation. This implementation keeps the proposed scientific contribution, dynamic region-concept-token reasoning with evidence traces and counterfactual probes, while adapting the data layer and terminology to retinal fundus imagery.

## What Is Included

- Region-aware retinal image encoder using fixed spatial patches.
- Concept vocabulary built from DeepEyeNet `Keywords`.
- Dynamic region-to-concept graph updated during decoding.
- LLM report decoders conditioned on the anatomy-aware explanation graph through learned soft-prefix embeddings, with both decoder-only and encoder-decoder HuggingFace models supported.
- Optional GRU decoder baseline with `--decoder-type gru`.
- Multi-task losses for report generation, concept prediction, graph alignment, sparsity, and temporal consistency.
- Concept coverage loss to reduce generic normal-report collapse by encouraging active findings to appear in the generated report.
- Evaluation suite for text quality, clinical concept fidelity, graph behavior, evidence faithfulness, and counterfactual sensitivity.
- Publication-oriented visualizations: metric bars, confusion heatmaps, graph diagrams, evidence heatmaps, counterfactual curves, and an interactive hover explanation viewer.
- A Colab notebook scaffold in `notebooks/DeepEyeNet_Dynamic_Graph_Colab.ipynb`.

## Expected Dataset Layout

Place or mount your Google Drive folder so it contains:

```text
DeepEyeNet/
  DeepEyeNet_train.json
  DeepEyeNet_valid.json
  DeepEyeNet_test.json
  train.csv
  valid.csv
  test.csv
  eyenet0420/
    train_set/
    val_set/
    test_set/
```

The JSON files should map image paths to:

```json
{
  "eyenet0420/train_set/example.jpg": {
    "Keywords": ["macular hole"],
    "clinical-description": "...",
    "report_text": "..."
  }
}
```

## Quick Start

DeepEyeNet:

```bash
pip install -r requirements.txt
python -m deepeyenet_dynamic_graph.train \
  --dataset deepeyenet \
  --data-root /path/to/DeepEyeNet \
  --output-dir outputs/run1 \
  --epochs 10 \
  --batch-size 8 \
  --decoder-type causal_lm \
  --llm-name distilgpt2 \
  --num-workers 0

python -m deepeyenet_dynamic_graph.evaluate \
  --dataset deepeyenet \
  --data-root /path/to/DeepEyeNet \
  --checkpoint outputs/run1/best_model.pt \
  --output-dir outputs/run1/eval \
  --num-workers 0
```

IU-XRay from Kaggle:

```bash
pip install -r requirements.txt
python -m deepeyenet_dynamic_graph.prepare_iuxray --output-dir outputs/iuxray_prepare
```

The script prints the Kaggle dataset path. Use that path as `--data-root`:

```bash
python -m deepeyenet_dynamic_graph.train \
  --dataset iuxray \
  --data-root /path/printed/by/prepare_iuxray \
  --output-dir outputs/iuxray_run1 \
  --epochs 10 \
  --batch-size 8 \
  --decoder-type causal_lm \
  --llm-name distilgpt2 \
  --num-workers 0

python -m deepeyenet_dynamic_graph.evaluate \
  --dataset iuxray \
  --data-root /path/printed/by/prepare_iuxray \
  --checkpoint outputs/iuxray_run1/best_model.pt \
  --output-dir outputs/iuxray_run1/eval \
  --split test \
  --num-workers 0
```

The evaluation folder includes `interactive_explanations.html`. Open it in a browser to hover over image regions and inspect anatomy, top findings, linked report text, and counterfactual drops.

## LLM Decoders

The default model is LLM-based. The explanation graph is converted into learned prefix embeddings:

```text
image patch -> anatomy node -> finding node -> LLM prefix/context -> generated report
```

During training and evaluation, the LLM hidden state at each token position recomputes the region-concept and token-concept edges. This keeps the explanation graph dynamic rather than broadcasting one static graph across the whole report.

Use decoder-only causal LMs such as GPT-2/BioGPT with:

```bash
--decoder-type causal_lm --llm-name distilgpt2
```

The older alias still works:

```bash
--decoder-type llm
```

Use encoder-decoder models such as T5/FLAN-T5 with:

```bash
--decoder-type seq2seq --llm-name google/flan-t5-small
```

The design choice is deliberate: causal LMs receive the graph as learned soft-prefix tokens before the report, while seq2seq LMs receive the graph as encoder context and generate from the decoder. Both paths return the same explanation tensors, so metrics, counterfactuals, and the interactive HTML viewer remain comparable across decoder families.

To make outputs less generic, training also includes a lightweight concept coverage objective. For each supervised finding, the model is penalized when the decoder never assigns high probability to the tokenizer pieces for that concept. The strength is controlled by:

```bash
--lambda-coverage 0.05
```

Evaluation reports `report_concept_mention_recall`, `report_concept_mention_precision`, `report_concept_mention_f1`, and `important_concept_omission_rate`, which are useful for catching cases where the generated report says only “no acute disease” while omitting important findings such as pneumothorax or pleural effusion.

## Concept Graphs

The graph concept vocabulary can be built from keywords, RadGraph-style entities, or a hybrid clinical extractor:

```bash
--concept-source hybrid
```

If you have RadGraph output JSON, pass:

```bash
--concept-source radgraph \
--radgraph-path /path/to/radgraph_outputs.json
```

To use cached LLM-assisted concept normalization, set `OPENAI_API_KEY` and add:

```bash
--concept-normalizer llm \
--concept-normalizer-model gpt-4o-mini
```

Training writes:

```text
concept_graph.json
concept_normalization_cache.json
```

You can inspect concept extraction separately:

```bash
python -m deepeyenet_dynamic_graph.prepare_concepts \
  --dataset iuxray \
  --data-root /path/to/IU-XRay \
  --output outputs/iuxray_concept_graph.json \
  --radgraph-path /path/to/radgraph_outputs.json
```

Default lightweight Colab setting:

```bash
--decoder-type causal_lm --llm-name distilgpt2
```

Stronger biomedical option:

```bash
--decoder-type causal_lm --llm-name microsoft/BioGPT
```

Instruction-tuned encoder-decoder option:

```bash
--decoder-type seq2seq --llm-name google/flan-t5-small
```

For a cheaper ablation, freeze the LLM and train only the image/graph/prefix modules:

```bash
--freeze-llm
```

For the older non-LLM baseline:

```bash
--decoder-type gru
```

## Ablation Examples

No anatomy layer:

```bash
python -m deepeyenet_dynamic_graph.train \
  --dataset iuxray \
  --data-root /path/to/IU-XRay \
  --output-dir outputs/iuxray_no_anatomy \
  --no-anatomy
```

No sparsity or temporal graph regularization:

```bash
python -m deepeyenet_dynamic_graph.train \
  --dataset iuxray \
  --data-root /path/to/IU-XRay \
  --output-dir outputs/iuxray_no_graph_reg \
  --lambda-sparse 0 \
  --lambda-temp 0
```

Different patch granularity:

```bash
python -m deepeyenet_dynamic_graph.train \
  --dataset deepeyenet \
  --data-root /path/to/DeepEyeNet \
  --output-dir outputs/deepeyenet_grid5 \
  --patch-grid 5
```

For Colab, open `notebooks/DeepEyeNet_Dynamic_Graph_Colab.ipynb`, mount Drive, set `DATA_ROOT`, and run the cells.

## Design Justification

The dataset is small, so the model uses a compact region encoder rather than a very large end-to-end transformer. This makes experiments feasible on Colab and reduces overfitting risk. Region nodes are fixed retinal patches, concept nodes come directly from the clinical keyword labels, and token nodes are generated report tokens. This design is faithful to the proposed methodology while keeping the implementation auditable and runnable with limited compute.

The training objective is deliberately decomposed. Report cross-entropy optimizes language generation; concept BCE encourages clinical label fidelity; weak graph alignment ties active concepts to keyword supervision; sparsity and temporal losses make explanations easier to inspect; faithfulness and counterfactual metrics are evaluated after training because they are more stable and cheaper as diagnostic probes than as heavy inner-loop objectives on a small dataset.

## Core Evaluation Metrics

- **Language quality:** BLEU-1 to BLEU-4, ROUGE-L, METEOR when available.
- **Clinical concept fidelity:** micro/macro precision, recall, F1, exact-match rate.
- **Graph quality:** edge entropy/sparsity, temporal graph drift, top-k concept hit rate.
- **Faithfulness:** confidence drop after masking top evidence regions.
- **Counterfactual sensitivity:** predicted concept probability reduction after targeted evidence suppression.

These metrics are complementary. Text metrics measure fluency and lexical overlap, concept metrics estimate clinical correctness, and explanation metrics test whether the graph evidence is sparse, stable, and causally useful.
