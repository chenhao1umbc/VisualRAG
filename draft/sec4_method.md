# Section 4: Method

## 4.1 Baseline Model

Our approach builds on **HunyuanOCR** (Tencent Hunyuan Vision Team, arXiv:2511.19575, 2025), a 1B-parameter end-to-end vision-language model designed for document OCR. The model consists of three components: a native-resolution ViT encoder (Hunyuan-ViT, 0.4B parameters) that processes input images at their original aspect ratio via adaptive patching; a learnable MLP connector that bridges visual and language representations; and a Hunyuan-0.5B LLM decoder that autoregressively generates the document transcription. The model is trained end-to-end with reinforcement learning and produces structured output in Markdown and HTML.

Out of the box, HunyuanOCR discards all text formatting attributes—strikethrough, underline, and text color are not represented in its output vocabulary, and its training data contains no annotation for these features. We adapt this model to recover all four target features (§3) using parameter-efficient fine-tuning, without modifying the base architecture or inference pipeline.

## 4.2 Dual LoRA Adaptation

We fine-tune HunyuanOCR using **Dual LoRA** (Xu et al., arXiv:2512.03402, 2025), a parameter-efficient adaptation method that decomposes weight updates into explicit magnitude and direction components. Dual LoRA adapters are applied to the attention and feed-forward layers of the LLM decoder; the ViT encoder weights are frozen in the primary configuration (see §5 for an ablation of ViT LoRA-tuning).

**Standard LoRA baseline.** Standard LoRA (Hu et al., ICLR 2022) modifies a frozen pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ by injecting a low-rank update:
$$W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} BA$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$. The rank constraint limits the expressivity of $\Delta W$: its effective rank is at most $r$, a fraction of the full-rank update achievable by unconstrained fine-tuning.

**Dual LoRA decomposition.** Dual LoRA uses four low-rank matrices divided into two groups. The **magnitude group** computes:
$$W_m = \operatorname{ReLU}(BA), \quad A \in \mathbb{R}^{r_1 \times k},\; B \in \mathbb{R}^{d \times r_1}$$
The ReLU nonlinearity produces non-negative outputs, which function as element-wise gates: parameters with large positive outputs are strongly updated, while negative pre-activations are zeroed out, effectively freezing well-adapted parameters. The **direction group** computes:
$$W_d = \operatorname{Sign}(DC), \quad C \in \mathbb{R}^{r_2 \times k},\; D \in \mathbb{R}^{d \times r_2}$$
The Sign function binarizes each element to $\{+1, -1\}$, controlling the direction (sign) of each parameter update independently of its magnitude. The combined weight update is:
$$\Delta W = \frac{\alpha}{\sqrt{r_1 r_2}}\, W_m \odot W_d = \frac{\alpha}{\sqrt{r_1 r_2}}\, \operatorname{ReLU}(BA) \odot \operatorname{Sign}(DC)$$
where $\odot$ denotes element-wise (Hadamard) product and $\frac{\alpha}{\sqrt{r_1 r_2}}$ normalizes the update scale across rank configurations.

**Gradient computation.** The Sign function has zero derivative almost everywhere, preventing standard backpropagation. Dual LoRA uses the Straight-Through Estimator (STE): during the backward pass, the gradient is passed through the Sign operation as an identity, clipped to $[-1, 1]$:
$$\frac{\partial \mathcal{L}}{\partial x} = \operatorname{Clip}\!\left(\frac{\partial \mathcal{L}}{\partial x_b},\, -1,\, 1\right)$$
This allows the direction matrices $C$ and $D$ to be trained end-to-end. Xu et al. note that standard LoRA's zero initialization is incompatible with Dual LoRA's ReLU gate: zeroing $A$ or $B$ would permanently suppress all gradients through the magnitude group. They state: "none of the low-rank matrices in the magnitude group should be initialized with zero... we use random Gaussian initialization for all four low-rank matrices and apply a warm-up strategy for the first few training steps to make sure that $\Delta W = 0$ at the start" (Xu et al., §3.3). We follow this initialization scheme in our experiments.

**Effective rank.** The Hadamard product of $\operatorname{ReLU}(BA)$ (effective rank $\leq r_1$) and $\operatorname{Sign}(DC)$ (effective rank $\leq r_2$) yields an update matrix with effective rank up to $r_1 \cdot r_2$. For our setting $r_1 = r_2 = r$, this gives up to $r^2$—far exceeding the rank-$r$ constraint of standard LoRA at the same parameter budget. This higher effective rank is the key mechanism by which Dual LoRA captures new visual and linguistic patterns—such as recognizing strikethrough line geometry, underline positioning, color-encoded text regions, and complex cell-merge structures—that fall outside the intrinsic dimensionality assumptions of standard LoRA.

**Comparison to DoRA.** DoRA (Liu et al., ICML 2024) also decomposes model parameters into magnitude and direction components, but operates on the *pre-trained weight* $W_0$: it rewrites $W_0 = m \cdot V / \|V\|$ (column magnitude times unit-direction matrix) and applies LoRA to directional updates of $V$. By contrast, Dual LoRA decomposes the *update* $\Delta W$, leaving $W_0$ intact and applying the magnitude/direction structure only to the fine-tuning delta. This distinction matters in practice: Dual LoRA does not require rewriting or re-normalizing the pre-trained weights, and its ReLU gate enables selective parameter freezing that DoRA does not provide.

In our primary configuration, we set $r_1 = r_2 = r$ to match the trainable parameter count of a standard LoRA adapter of rank $2r$, enabling a direct fair comparison in the ablation study (§5).

## 4.3 Multi-Feature Training Objective

A single Dual LoRA-augmented HunyuanOCR model is trained jointly on all four formatting features—strikethrough, underline, highlighted text, and complex financial tables—rather than training four separate specialized models. Joint training is motivated by the observation that financial documents routinely contain multiple formatting features on the same page, and a deployed system must handle arbitrary combinations at inference time without feature-detection pre-processing.

**Loss function.** Training uses standard autoregressive cross-entropy loss over ground truth output tokens with teacher forcing. No task-specific heads or auxiliary losses are added; the model learns to produce the correct output format string (`~~text~~`, `<u>text</u>`, color spans, or structural HTML) as part of the natural language generation objective. The output format is determined entirely by the input image, with no explicit feature-selector prompt.

**Replay buffer.** To mitigate catastrophic forgetting of general OCR capabilities during domain-specific fine-tuning, each training batch includes 10–15% samples drawn from the SynFinTabs general replay buffer (5,000 samples from the SynFinTabs train split; §3.2). These samples cover a diverse range of financial document layouts, including plain text, tables without merged cells, and mixed-content pages, maintaining the model's baseline OCR performance on inputs that do not contain the four target features.

**FinTabNet.c usage.** FinTabNet.c (Smock et al., ICDAR 2023) is used exclusively for **evaluation** of baseline vanilla HunyuanOCR on complex table recognition (project plan §1.1); it is not included in the training split. This ensures that the complex table evaluation metric (TEDS; Zhong et al., ECCV 2020) is measured on a held-out benchmark with no distribution overlap with the EDGAR training tables.

## 4.4 Output Format and Decoding

The model is trained to produce a unified output representation that extends standard Markdown with minimal, well-scoped inline HTML elements. No new special tokens are added to the tokenizer vocabulary; all formatting constructs are expressed through sequences drawn from the standard HunyuanOCR tokenizer, which natively represents HTML and Markdown syntax. The target output formats are:

- **Strikethrough**: `~~deleted text~~` — two tilde characters on each side, consistent with CommonMark Markdown extension syntax
- **Underline**: `<u>underlined text</u>` — standard HTML inline element
- **Highlighted text**: `<span style="background-color:NAME;">text</span>` — HTML inline element with a named color from the closed vocabulary {`red`, `blue`, `green`, `orange`, `purple`, `yellow`, `gray`}
- **Complex tables**: Full structural `<table>` HTML with `colspan` and `rowspan` attributes preserved; all presentational attributes stripped
- **Plain text and formulas**: Standard Markdown and `$$...$$` LaTeX, unchanged from vanilla HunyuanOCR output

At inference, the model performs a single autoregressive decoding pass over the full input image. No post-processing stage is applied; the formatting annotations are embedded directly in the output token stream. The closed named-color vocabulary simplifies color recognition to a seven-class classification problem embedded within the generation task, avoiding the need for fine-grained color regression or a separate colorimetry module.
