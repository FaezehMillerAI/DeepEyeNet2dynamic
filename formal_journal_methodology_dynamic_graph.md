# Formal Journal-Style Methodology Subsection

## 3 Methodology

This section presents a revised explainable medical image captioning framework in which the original Medical Language Graph is extended into a dynamic explanation graph. Unlike conventional post-hoc explainability methods that analyze model behavior after prediction, the proposed approach embeds explainability within the report generation process itself by jointly modeling region-level visual evidence, concept-level medical knowledge, token-level report generation, and counterfactual reasoning [cite:1][cite:2]. The framework is designed to produce a medical report together with an interpretable reasoning trace that can be audited at the level of image regions, medical concepts, and generated textual findings [cite:1][cite:2][cite:3].

### 3.1 Problem Statement

Let the input chest X-ray image be denoted by \(X\), and let the target medical report be represented by a sequence of tokens \(Y = \{y_1, y_2, \ldots, y_T\}\), where \(T\) is the report length. The objective is not only to generate a clinically relevant report from the medical image, but also to produce an interpretable evidence structure that explains how the report is formed. To achieve this, the proposed model introduces a time-dependent explanation graph \(G_t\), which evolves during decoding and captures the relationships among localized image regions, medical concepts, and generated tokens.

The overall conditional formulation can be expressed as

\[
P(Y, G_{1:T} \mid X) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, G_t, X) \, P(G_t \mid G_{t-1}, X, y_{<t}). \tag{1}
\]

Here, \(G_t\) represents the explanation graph at decoding step \(t\), and \(y_{<t}\) denotes the previously generated tokens. This formulation differs from the original paper, where the graph acts mainly as a representation layer and explainability is subsequently examined using LRP and LIME as separate post-hoc tools [cite:1]. In the revised formulation, explanation generation is incorporated directly into the predictive mechanism, such that each generated token can be associated with an explicit evidence path.

### 3.2 Dynamic Explanation Graph

The proposed Dynamic Explanation Graph is constructed from three interacting node types: region nodes, concept nodes, and token nodes. Region nodes correspond to localized image regions extracted from the chest X-ray, concept nodes represent clinically meaningful medical findings, and token nodes correspond to generated report words or subword units. At decoding step \(t\), the graph is defined as

\[
G_t = (V_r \cup V_c \cup V_y, E_{rc}^{(t)} \cup E_{cy}^{(t)} \cup E_{ry}^{(t)}), \tag{2}
\]

where \(V_r\), \(V_c\), and \(V_y\) denote the sets of region, concept, and token nodes, respectively. The edge sets \(E_{rc}^{(t)}\), \(E_{cy}^{(t)}\), and \(E_{ry}^{(t)}\) denote the dynamic relations among regions and concepts, concepts and tokens, and regions and tokens.

#### 3.2.1 Region-aware visual representation

To focus the model on diagnostically relevant evidence, the input image \(X\) is first processed by a segmentation network to obtain a set of localized regions of interest:

\[
R = \{r_1, r_2, \ldots, r_N\}. \tag{3}
\]

Each region \(r_i\) is then encoded using a Vision Transformer, yielding

\[
\mathbf{h}_i = f_{\mathrm{ViT}}(r_i), \quad i = 1, \ldots, N. \tag{4}
\]

To preserve spatial and anatomical information, the region representation is enriched with positional and anatomical priors:

\[
\tilde{\mathbf{h}}_i = \mathbf{h}_i + \mathbf{p}_i + \mathbf{a}_i, \tag{5}
\]

where \(\mathbf{p}_i\) and \(\mathbf{a}_i\) denote positional and anatomical embeddings, respectively. This formulation allows the model to distinguish between visually similar patterns occurring in different clinically meaningful locations.

#### 3.2.2 Graph initialization and update

The initial graph \(G_0\) is constructed from data-driven medical term statistics and prior domain knowledge. Specifically, concept nodes are initialized from medical terms extracted from the training corpus, while prior edges may be regularized using external medical relations or ontology-level knowledge [cite:1][cite:3]. However, unlike a fixed graph, the proposed graph is updated dynamically at each decoding step according to current image evidence, decoder state, and predicted clinical concepts:

\[
G_t = \phi(G_{t-1}, R, \mathbf{s}_{t-1}, \hat{\mathbf{c}}_{t-1}). \tag{6}
\]

Here, \(\phi(\cdot)\) denotes the graph update function, \(\mathbf{s}_{t-1}\) is the decoder hidden state, and \(\hat{\mathbf{c}}_{t-1}\) represents the predicted concept state from the previous step. This design enables the graph to adapt its relational structure as the report evolves, rather than relying on a static dependency pattern for all samples.

#### 3.2.3 Dynamic edge estimation

The region-to-concept relevance is calculated using an attention mechanism:

\[
\alpha_{ij}^{(t)} = \frac{\exp\left((W_r \tilde{\mathbf{h}}_i)^\top (W_c \mathbf{m}_j)\right)}{\sum_{j'} \exp\left((W_r \tilde{\mathbf{h}}_i)^\top (W_c \mathbf{m}_{j'})\right)}, \tag{7}
\]

where \(\mathbf{m}_j\) denotes the embedding of medical concept \(c_j\). Similarly, concept-to-token relevance is estimated by

\[
\beta_{jk}^{(t)} = \frac{\exp\left((U_c \mathbf{m}_j)^\top (U_y \mathbf{w}_k)\right)}{\sum_{k'} \exp\left((U_c \mathbf{m}_j)^\top (U_y \mathbf{w}_{k'})\right)}, \tag{8}
\]

where \(\mathbf{w}_k\) is the embedding of token \(y_k\). These dynamically updated edge weights define how image evidence is propagated toward medical concepts and finally translated into report tokens.

#### 3.2.4 Graph propagation

To integrate region evidence, concept interactions, and token context, message passing is performed on the graph as follows:

\[
\mathbf{z}_v^{(t+1)} = \sigma \left( \sum_{u \in \mathcal{N}(v)} \gamma_{uv}^{(t)} W^{(t)} \mathbf{z}_u^{(t)} \right), \tag{9}
\]

where \(\mathbf{z}_v^{(t)}\) is the hidden representation of node \(v\), \(\mathcal{N}(v)\) denotes its neighborhood, and \(\gamma_{uv}^{(t)}\) is the dynamic edge weight. Through this mechanism, the graph becomes a structured reasoning space in which visual, conceptual, and linguistic evidence interact throughout report generation.

### 3.3 Medical Language Decoder

The decoder generates the medical report token by token while conditioning on the current graph state. Let \(\mathbf{s}_t\) denote the hidden state of the decoder at time step \(t\). Then,

\[
\mathbf{s}_t = f_{\mathrm{dec}}(\mathbf{s}_{t-1}, y_{t-1}, G_t). \tag{10}
\]

The probability of generating the next token is then given by

\[
P(y_t \mid y_{<t}, G_t, X) = \mathrm{Softmax}(W_o \mathbf{s}_t). \tag{11}
\]

In contrast to a conventional decoder that attends only to image features or a fixed graph representation, the proposed decoder conditions on the dynamically updated explanation graph at each time step. Consequently, the generated sentence is not only conditioned on previously generated words, but also on the current evidence structure that encodes which regions and concepts remain most relevant.

To explicitly capture interpretability, the model computes a sparse evidence trace for each token:

\[
\mathbf{e}_t = \mathrm{TopK}\big(\mathrm{Attn}(\mathbf{s}_t, [R; G_t])\big). \tag{12}
\]

Here, \(\mathbf{e}_t\) denotes the minimal supporting evidence set for token \(y_t\), selected from both image regions and graph nodes. Sentence-level explanation is obtained by aggregating token-level evidence traces across all tokens within the sentence. This allows each generated finding to be traced back to specific image regions and concept nodes rather than being justified only by post-hoc saliency maps.

### 3.4 XAI Module

The explainability component of the proposed framework is designed as an integrated module rather than an isolated post-hoc analysis stage. It consists of two complementary elements: evidence-consistent explanation tracing and counterfactual explanation generation.

#### 3.4.1 Evidence-consistent tracing

For each generated token or sentence, the model identifies the corresponding supporting regions and concepts through the evidence trace in Equation (12). In this way, the explanation follows a structured path from localized image evidence to medical concept activation and finally to textual report generation. This is stronger than the original LRP-LIME combination because it provides a unified explanation chain rather than separate image and text interpretations [cite:1][cite:2].

To ensure that the selected evidence is faithful, an evidence-consistency objective is introduced so that removing highly ranked evidence decreases the confidence of the associated token. This can be expressed as

\[
\mathcal{L}_{cons} = \sum_{t=1}^{T} \max\left(0, m - \big(p(y_t \mid E_t) - p(y_t \mid \bar{E}_t)\big)\right), \tag{13}
\]

where \(E_t\) is the selected evidence set, \(\bar{E}_t\) is its complement, and \(m\) is a predefined margin. This encourages the model to rely on genuinely causal evidence rather than visually plausible but non-essential features.

#### 3.4.2 Counterfactual explanation generation

To further enhance interpretability, a counterfactual branch is introduced to identify the minimal evidence change that would alter a predicted finding. Let \(q\) denote a target clinical finding. The optimal perturbation is defined as

\[
\delta^* = \arg\min_{\delta} \left( \|\delta\|_2 + \lambda \mathcal{L}_{flip}(q; X + \delta) + \eta \mathcal{L}_{preserve}(Y_{\setminus q}) \right). \tag{14}
\]

Here, \(\mathcal{L}_{flip}\) enforces a change in the selected finding, while \(\mathcal{L}_{preserve}\) preserves the remaining report content. This enables explanations of the form: the model predicts opacity because of evidence in particular regions and concept nodes, and if that evidence were reduced, the predicted finding would change. Such a mechanism provides a clinically meaningful notion of explanation by clarifying not only why the prediction occurred, but also what evidence would be required for a different outcome [cite:2].

#### 3.4.3 Joint learning objective

The complete training loss combines report generation, graph alignment, explanation sparsity, temporal graph consistency, evidence faithfulness, and counterfactual validity:

\[
\mathcal{L} = \mathcal{L}_{rep} + \lambda_1 \mathcal{L}_{graph} + \lambda_2 \mathcal{L}_{align} + \lambda_3 \mathcal{L}_{sparse} + \lambda_4 \mathcal{L}_{temp} + \lambda_5 \mathcal{L}_{cons} + \lambda_6 \mathcal{L}_{cf}. \tag{15}
\]

The temporal consistency term constrains the graph to evolve smoothly during report decoding:

\[
\mathcal{L}_{temp} = \sum_{t=2}^{T} \| G_t - \Psi(G_{t-1}) \|_1, \tag{16}
\]

where \(\Psi(\cdot)\) is a regularization operator on the graph update. This prevents unstable or overly noisy graph transitions and improves the interpretability of the evolving reasoning process.

### 3.5 Inference and Explanation Output

During inference, the model produces four synchronized outputs for each input chest X-ray: (1) the generated medical report, (2) the supporting image regions for each sentence, (3) the dynamic graph path connecting regions, concepts, and tokens, and (4) a counterfactual explanation indicating the minimal evidence change required to alter a selected finding. As a result, the proposed system functions not only as a report generator but also as an interpretable clinical reasoning model.

### 3.6 Methodological Novelty

The key methodological novelty of the proposed framework lies in replacing a static medical language graph with a dynamic explanation graph that evolves during report generation. Unlike the original approach, which applies LRP and LIME as separate post-hoc explainability tools [cite:1], the revised framework treats explainability as a native property of the generation process by jointly modeling region-level grounding, concept-level reasoning, token-level evidence tracing, temporal graph updates, and counterfactual analysis within a single end-to-end architecture. This produces explanations that are more faithful, more structured, and more clinically auditable than conventional saliency-based interpretations [cite:2][cite:3].

A concise paper-ready novelty statement is as follows:

> This work introduces a dynamic evidence-consistent explanation graph for medical image captioning, in which region-concept-token relations are updated throughout report generation. The proposed framework jointly performs grounded report decoding, evidence tracing, temporal graph reasoning, and counterfactual explanation within a single end-to-end model, thereby moving beyond static post-hoc saliency toward adaptive and clinically traceable explainability [cite:1][cite:2][cite:3].
