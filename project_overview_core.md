# Project Overview (Core — Do Not Change)

## Goal

Build a specialized OCR model for financial documents that correctly handles:
1. **Strikethrough text** — currently ignored by all mainstream OCR models
2. **Underline text** — currently ignored by all mainstream OCR models
3. **Complex financial tables** — nested headers, merged cells, multi-level structures that general OCR fails on
4. **Colored text** — color carries semantic meaning in financial documents, currently discarded by all OCR models

Target output: a conference paper in the OCR/document understanding domain.

---

## Motivation

General-purpose OCR models (including strong baselines like HunyuanOCR) ignore strikethrough, underline, and text color entirely, and fail on complex financial table structures. In financial documents, these are not decorative — they carry semantic meaning:
- Strikethrough indicates deleted/amended values (common in regulatory filings, contracts, term sheets)
- Underline indicates emphasis or totals (common in financial statements)
- Complex tables with merged cells and multi-level headers are standard in balance sheets, segment reporting, and SEC filings
- Color encodes semantic information: red = negative values/warnings, blue = hyperlinks/references, yellow/green = highlighted key figures, colored headers = category groupings

No existing dataset or model specifically targets all four of these features together in the financial domain.

---

## Baseline Model

**HunyuanOCR** (`tencent/HunyuanOCR`)
- Architecture: HunYuanVLForConditionalGeneration (VLM = ViT + Hunyuan-0.5B LLM)
- Inference: via HuggingFace transformers (`run_hy_ocr.py`)
- Training: standard LoRA already implemented (`finetune_lora.py`)
- Current LoRA targets: LLM layers only (`model.layers.*`), ViT frozen

---

## Training Method

**Dual LoRA** — SOTA LoRA variant that decomposes weight updates into:
- **Magnitude** component (scaling)
- **Direction** component (rotation)

Reference: *Dual LoRA: Enhancing LoRA with Magnitude and Direction Updates* (arXiv 2512.03402)

Rationale: Direction updates can capture new visual/semantic representations (strikethrough lines, underline marks, complex table structures) while magnitude updates preserve existing OCR knowledge. This is the technical novelty angle vs. vanilla LoRA.

---

## Output Format Conventions

| Feature | Ground Truth Format |
|---|---|
| Strikethrough | `~~text~~` (markdown) |
| Underline | `<u>text</u>` (HTML inline) |
| Colored text | `<span style="color:NAME;">text</span>` (HTML inline, named colors only) |
| Complex tables | `<table>` with `colspan`/`rowspan` (HTML) |
| Regular text | Markdown |
| Formulas | LaTeX (`$$...$$`) |

Named color set (financial palette): `red`, `blue`, `green`, `orange`, `purple`, `yellow`, `gray`
Black/default text requires no color annotation.

---

## Research Gap Confirmed

- Only one prior paper addresses strikethrough OCR: *"An Approach of Strike-Through Text Identification from Handwritten Documents"* (IEEE 2014) — handwritten, not financial, not VLM-based
- OmniDocBench (CVPR 2025) covers financial documents but does **not** annotate strikethrough, underline, or text color
- No existing benchmark covers all four target features in the financial domain together
- This confirms the novelty of both the dataset and the model contribution

---

## Paper Contribution Structure (Proposed)

1. **FinDocOCR Dataset** — first dataset targeting strikethrough, underline, colored text, and complex tables in financial documents (to be released)
2. **Model** — HunyuanOCR + Dual LoRA fine-tuned on FinDocOCR
3. **Evaluation** — demonstrate baseline failure and model improvement on all four features
