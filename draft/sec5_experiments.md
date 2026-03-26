# Section 5: Experiments

## 5.1 Experimental Setup

**Hardware.** All training runs are conducted on NVIDIA A100 80GB GPUs [TBD: count]. Mixed-precision training (bfloat16) is used throughout to reduce memory footprint while maintaining numerical stability.

**Optimizer.** We use AdamW with learning rate [TBD] and weight decay [TBD]. Dual LoRA supports separate learning rates for the magnitude group ($A$, $B$) and direction group ($C$, $D$); we sweep the magnitude-to-direction LR ratio over $\{0.1, 1.0, 10.0\}$ and report the best configuration [TBD]. A linear warm-up over the first 50 steps scales the adapter contribution from zero to its full value, consistent with Xu et al.'s initialization protocol (§3.3 of arXiv:2512.03402). Batch size: [TBD] images per GPU, with gradient accumulation to an effective batch size of [TBD].

**Dual LoRA rank.** We sweep $r_1 = r_2 = r \in \{8, 16, 32\}$ and select the best rank configuration based on validation loss on the held-out 200-page evaluation set. The reported main result uses $r = $ [TBD]. For the standard LoRA baseline, we use rank $2r$ to match the trainable parameter count, ensuring a fair comparison.

**Training duration.** All models are trained for [TBD] epochs. Dual LoRA adapters are applied to the attention and feed-forward layers of the Hunyuan-0.5B LLM decoder; the ViT encoder is frozen in the primary configuration (§4.2). The standard LoRA baseline uses the same target layers and rank budget.

**Evaluation protocol.** All models are evaluated on the held-out 200-page set (§3.4), with 50 pages per feature category (strikethrough, underline, highlighted text, complex tables). Evaluation metrics are:

- **Strikethrough F1**: Character-level F1 between predicted and ground-truth `~~...~~` spans. Precision and recall are computed over the set of character positions predicted as strikethrough vs. those in the ground-truth annotation; span boundaries are derived by matching `~~...~~` delimiters in the predicted string against the ground-truth annotation.
- **Underline F1**: Character-level F1 over predicted `<u>...</u>` spans, computed analogously; span boundaries are derived by matching `<u>...</u>` delimiters in the predicted string against the ground-truth annotation.
- **Color Accuracy**: Per-token classification accuracy over the seven-class closed color vocabulary (`red`, `blue`, `green`, `orange`, `purple`, `yellow`, `gray`), evaluated only on tokens within `<span style="background-color:...">` regions.
- **Table TEDS**: Tree Edit Distance Similarity (Zhong et al., ECCV 2020) between predicted and ground-truth `<table>` HTML, capturing both cell content and merge structure (`colspan`/`rowspan`). Evaluated on the 50 complex-table pages.
- **Overall**: Macro-average of the four per-feature metrics. Although these metrics measure different phenomena (span detection, classification, and structural similarity), all are normalized to [0, 1], making their average a meaningful summary of a model's across-the-board formatting capability.

---

## 5.2 Main Results

Table 1 presents the main comparison of vanilla HunyuanOCR, HunyuanOCR with standard LoRA fine-tuning, and our proposed HunyuanOCR with Dual LoRA fine-tuning. All fine-tuned models are trained on the full FinDocOCR training set with the 10–15% SynFinTabs replay buffer (§4.3).

**Table 1: Main results on the FinDocOCR held-out evaluation set.**

| Model | Strikethrough F1 | Underline F1 | Color Acc. | Table TEDS | Overall |
|---|---|---|---|---|---|
| HunyuanOCR (vanilla) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + Standard LoRA (rank $2r$) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + Dual LoRA, ours (rank $r$) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Tan et al. (2025)† | — | — | — | [TBD] | — |

† Tan et al. (arXiv:2508.05669) fine-tune Qwen2.5-VL-7B with standard LoRA on financial table conversion; their evaluation covers only the table recognition feature on a different (Malaysian financial report) dataset and is not directly comparable. Table TEDS is reproduced here for reference only; no cross-dataset comparison is implied.

Vanilla HunyuanOCR is expected to score near zero on strikethrough, underline, and color (these features are absent from its output vocabulary), and to achieve [TBD] TEDS on complex tables (reflecting its general HTML parsing capability without merge-structure training). Both LoRA variants are expected to show substantial improvement across all four features after fine-tuning on FinDocOCR.

---

## 5.3 Ablation Study

### Ablation 1 — ViT Encoder Tuning

Our primary configuration freezes the ViT encoder and applies Dual LoRA adapters only to the LLM decoder (§4.2). To assess whether visual-layer adaptation is beneficial, we additionally train a configuration in which Dual LoRA adapters are applied to both the LLM decoder and the ViT encoder (LLM+ViT).

**Table 2: Ablation — LLM-only vs. LLM+ViT Dual LoRA.**

| Configuration | Strikethrough F1 | Underline F1 | Color Acc. | Table TEDS | Overall |
|---|---|---|---|---|---|
| LLM-only (primary) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| LLM+ViT | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| $\Delta$ (LLM+ViT − LLM-only) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

**Hypothesis.** ViT encoder adaptation is expected to improve performance on the three visual formatting features (strikethrough, underline, color), where recognizing geometric cues—a horizontal strikethrough line, the position of an underline mark, color regions—requires visual representations that the frozen ViT may not encode optimally. For complex tables, ViT tuning may offer marginal benefit since structural parsing is primarily a language-model task. A risk of ViT tuning is interference with the encoder's general document layout understanding, which could degrade TEDS on structurally simple tables.

### Ablation 2 — Replay Buffer

To quantify the effect of catastrophic forgetting mitigation, we compare the primary Dual LoRA model (with 10–15% SynFinTabs replay per batch) against an otherwise identical model trained without any replay samples.

**Table 3: Ablation — replay buffer effect on general OCR.**

| Configuration | Strikethrough F1 | Underline F1 | Color Acc. | Table TEDS | CER (general OCR)† |
|---|---|---|---|---|---|
| With replay (primary) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Without replay | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

† CER measured on the OmniDocBench general OCR subset to assess regression on documents without formatting features. The success criterion for the replay buffer is that CER/WER degradation relative to vanilla HunyuanOCR does not exceed 5%.

**Hypothesis.** Without the replay buffer, the model is expected to overfit to the FinDocOCR feature distribution and degrade on plain-text financial pages. The replay buffer is expected to keep general OCR CER within 5% of the vanilla baseline while preserving feature-specific gains.

---

## 5.4 Error Analysis

Based on dataset characteristics and the structure of the target output formats, we anticipate several recurring failure modes once results are available. Short strikethrough spans (one or two characters) are likely to be confused with hyphenation or em-dashes, since the visual signal of a short horizontal line may be ambiguous at the patch level. Underline detection is expected to be sensitive to font size: in small-font footnote text, the underline mark falls within the same pixel row as the descenders, reducing the geometric separation cue available to the ViT encoder. For highlighted text, low-saturation colors—particularly gray and pale shades that map to `yellow` or `orange` in the closed vocabulary—are expected to show the highest misclassification rate, as the color signal is attenuated relative to high-saturation primaries. For complex tables, preliminary inspection of the EDGAR training data suggests that merge structures deeper than three levels of `colspan`/`rowspan` nesting are underrepresented; we expect TEDS to degrade on tables with nested headers beyond this depth. These hypotheses will be validated by examining per-sample TEDS and span-level F1 breakdowns in the final evaluation.
