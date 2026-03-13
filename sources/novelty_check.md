# Task 1.0 — Novelty Verification

**Claim**: No prior work addresses all four features — strikethrough OCR, underline OCR, colored-text OCR, and complex financial table OCR — together, in a financial document context, using a VLM-based approach.

**Search date**: 2026-03-11
**Searches conducted**: "strikethrough OCR financial documents", "underline text detection OCR", "colored text OCR semantic financial", "OmniDocBench annotations", "financial table recognition VLM merged cells", "Dual LoRA document OCR fine-tuning"

---

## Feature (a): Strikethrough OCR in Typed/Printed Financial Documents

### Papers found:

**Adak & Chaudhuri (2014)** — "An Approach of Strike-Through Text Identification from Handwritten Documents"
_IEEE ICFHR 2014, pp. 643–648_
- Addresses: binary detection of struck-out text (yes/no per word) in **handwritten** documents
- Scope gap: handwritten manuscripts only; no typed/printed document support; no financial domain; no output of the struck-out text content; binary classification not OCR-with-formatting
- Method: classical image processing (not VLM, not deep learning)

**Poddar et al. (2021)** — "Detection and Localisation of Struck-Out-Strokes in Handwritten Manuscripts"
_Arnab Poddar, Akash Chakraborty, Jayanta Mukhopadhyay, Prabir Kumar Biswas_
_ICDAR 2021 Workshops (Document Analysis and Recognition), Lecture Notes in Computer Science vol. 12917, pp. 98–112, Springer, Lausanne, Switzerland, September 2021. DOI: 10.1007/978-3-030-86159-9\_7_
- Addresses: stroke-level localization in **handwritten historical manuscripts**
- Scope gap: same as above — handwritten, not printed, not financial, no semantic output

### Gap confirmed:
No paper addresses strikethrough in typed/printed financial documents. No paper outputs `~~text~~` (the struck-through content itself) for financial filings. The 2014 paper is the only prior strikethrough work and is fundamentally out of scope on every dimension: document type, domain, and task formulation.

---

## Feature (b): Underline OCR — Underlined Text Detection and Output

### Papers found:
None. Searches for "underline text detection OCR", "underline recognition document understanding" returned no papers specifically targeting underline as a distinct recognized formatting feature in document OCR output. All OCR models surveyed (HunyuanOCR, GOT-OCR2, olmOCR, Mistral OCR) discard underline formatting in their output.

### Gap confirmed:
No prior work outputs underline annotations (`<u>text</u>`) as part of OCR recognition for financial or general documents.

---

## Feature (c): Colored Text OCR — Semantic Color Extraction

### Papers found:

**Hase et al. (2003)** — "Color segmentation for text extraction"
_H. Hase, M. Yoneda, S. Tokai, J. Kato, C. Y. Suen_
_International Journal on Document Analysis and Recognition (IJDAR), vol. 6, pp. 271–284, Springer, 2003. DOI: 10.1007/s10032-003-0119-7_
- Addresses: using color channels to improve character segmentation for legacy OCR systems
- Scope gap: this is pre-deep-learning color channel selection to improve **character recognition accuracy**, not semantic color annotation as output. Does not identify or output the color of text as part of document understanding.

No modern VLM-based paper outputs the semantic color of text (e.g., `<span style="color:red;">...</span>`) as part of document parsing. OmniDocBench (CVPR 2025) does not annotate text color at the span level (confirmed below).

### Gap confirmed:
No paper models semantic text color extraction as an OCR output feature in financial documents.

---

## Feature (d): Complex Financial Table OCR with Merged Cells

### Papers found (partial coverage — none is our work):

**Zheng et al., FinTabNet (WACV 2021)**: Cell-structure annotations for financial tables; focuses on bounding-box cell detection. Does **not** use VLMs; does not address strikethrough/underline/color.

**Smock et al., PubTables-1M (CVPR 2022)**: Large-scale table detection and structure recognition; scientific/biomedical domain, not financial. Does not address the other three features.

**Smock et al., FinTabNet.c (ICDAR 2023)**: Curated/corrected version of FinTabNet annotations. Cell-level annotations only; no VLM; no other features.

**Bradley et al., SynFinTabs (arXiv 2412.04262, 2024)**: Synthetic financial table images with HTML/JSON/CSV annotations. Single-feature (tables only); no strikethrough/underline/color.

**Tan et al. (2025)** — "Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports" (arXiv:2508.05669): Fine-tuning Qwen2.5-VL-7B with LoRA on 2,152 image-text pairs for financial table Markdown extraction; achieves 96.53% TEDS. Single-feature (tables only); no strikethrough/underline/color.

### Gap: financial tables + VLM is partially addressed, but always in isolation from the other three features.

---

## OmniDocBench (CVPR 2025) — Annotation Schema Verified

**Source**: Official GitHub README (opendatalab/OmniDocBench), fetched 2026-03-11

**Block-level annotations (15 types)**: title, text_block, figure, figure_caption, figure_footnote, table, table_caption, table_footnote, equation_isolated, equation_caption, header, footer, page_number, page_footnote, abandon, code_txt, code_txt_caption, reference

**Span-level annotations (4 types)**: text_span, equation_ignore, equation_inline, footnote_mark

**Confirmed**: OmniDocBench does **NOT** include strikethrough, underline, or colored text as annotation categories. These formatting attributes are entirely absent from the benchmark's annotation schema.

---

## Dual LoRA + Document OCR Fine-Tuning

**Search**: "Dual LoRA document OCR", "Dual LoRA vision language model fine-tuning"

No paper found combining Dual LoRA (arXiv 2512.03402) with document OCR or vision-language model fine-tuning. The Dual LoRA paper itself experiments only on NLP tasks (NLU, commonsense reasoning) using RoBERTa, DeBERTa, and LLaMA models. Its application to VLM-based document understanding is novel.

---

## Summary Table

| Feature | Prior work? | Closest paper | Gap |
|---|---|---|---|
| Strikethrough OCR (printed, financial) | No | Adak & Chaudhuri IEEE 2014 | Handwritten only; binary detection; no financial domain |
| Underline OCR | No | None found | Completely unaddressed |
| Colored text OCR (semantic) | No | IJDAR 2003 (color channel selection) | Not semantic output; pre-deep-learning |
| Financial table OCR (VLM) | Partial | arXiv 2508.05669 (2025) | Single feature only; no other three features |
| All four features combined | No | None found | Our work is first |
| OmniDocBench coverage | N/A | OmniDocBench CVPR 2025 | Does not annotate strikethrough/underline/color |
| Dual LoRA for document VLM | No | Xu et al. arXiv 2512.03402 | Only NLP tasks; never applied to OCR or VLMs |

---

## Confidence Assessment

**High confidence** that no paper addresses all four features together. Gap on strikethrough (printed) and underline is extremely clear — no papers found. Gap on colored text OCR is clear for semantic output. Financial table + VLM is partially covered but only in isolation.

**One honest caveat**: We did not exhaustively search Chinese-language venues (CCIR, VALSE, etc.) where HunyuanOCR and similar work originates. It is possible a Chinese-language paper addresses one of these features. However, no such paper appears in arXiv, IEEE Xplore, or ACM DL searches.
