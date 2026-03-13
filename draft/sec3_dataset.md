# Section 3: The FinDocOCR Dataset

## 3.1 Overview and Design Principles

We construct **FinDocOCR**, a multi-feature dataset for format-aware financial document OCR. The dataset is designed around four formatting features that carry semantic meaning in financial documents but are discarded by all existing OCR systems: **strikethrough text** (indicating deleted or amended values), **underlined text** (indicating emphasis or running totals), **colored text** (encoding semantic categories such as warnings, hyperlinks, or key figures), and **complex financial tables** with merged cells and multi-level headers.

The design is motivated by two observations from the related work survey. First, no existing benchmark—including OmniDocBench (Ouyang et al., CVPR 2025), the most comprehensive open benchmark for document parsing—provides annotation types for strikethrough, underline, or text color (§2.2). Second, financial document AI research has established that structural and semantic richness in financial filings creates distinct OCR challenges that general-purpose datasets do not address (Chen et al., EMNLP 2021; Zhu et al., ACL 2021). FinDocOCR is the first dataset to combine all four target features in a unified training and evaluation framework.

All data sources are combined into a single annotation schema (§3.3), and the dataset is divided into a training split used for fine-tuning and a held-out evaluation set with human annotation (§3.4).

## 3.2 Data Sources

FinDocOCR draws from three complementary sources, each contributing distinct coverage of the target features.

**EDGAR Financial Filings.** We build a custom pipeline over the SEC EDGAR public filing system, querying 10-K, 10-Q, and amendment filings (10-K/A, DEF 14A, S-1/A) via the EDGAR EFTS full-text search API. For complex tables, HTML tables are extracted from filing documents, filtered to retain only tables with at least 3 rows, at least 2 columns, and at least one merged cell (colspan > 1 or rowspan > 1). Ground truth is the sanitized structural HTML with all XBRL namespaced tags unwrapped and all inline CSS stripped, preserving only `colspan` and `rowspan` attributes. Images are rendered from the HTML using a PIL-based renderer. This pipeline produces **1,301 complex table pairs** from 500 filings.

For strikethrough, we filter the same EDGAR corpus for amendment-type filings, which contain `<del>` tags and `text-decoration: line-through` CSS spans indicating struck-through content. Spans with fewer than 10 characters are discarded as noise. This produces **446 strikethrough pairs** from real SEC amendment documents. Ground truth uses the `~~text~~` markdown convention.

**Synthetic Generator.** To supplement the relatively small real-world counts for strikethrough, underline, and color, we build a unified synthetic document generator that renders financial text content (drawn from EDGAR sample sentences) onto document images with programmatically controlled formatting. The generator varies font, size, line thickness, and background to improve robustness:
- *Strikethrough*: single and double strike, varying thickness; **5,000 pairs**
- *Underline*: single and double underline, varying baseline offset; **5,000 pairs**
- *Color*: seven named colors from the financial palette (`red`, `blue`, `green`, `orange`, `purple`, `yellow`, `gray`) at varying weights and backgrounds; **5,000 pairs** plus **5,000 mixed-feature pairs** where multiple formatting attributes co-occur on the same page

**SynFinTabs (Bradley et al., 2024).** We use the SynFinTabs dataset (arXiv:2412.04262) from HuggingFace (`ethanbradley/synfintabs`) in two roles. A 200-sample SynFinTabs test subset is used for **pre-fine-tuning baseline evaluation** of vanilla HunyuanOCR on complex table recognition; these samples are also included in the complex table training count in §3.5 (no data leakage: the baseline is measured before any fine-tuning). An additional 5,000 samples drawn from the training split serve as a **general replay buffer**, providing typographic and layout diversity beyond the EDGAR-domain training data. This replay buffer is mixed into every training batch at a 10–15% rate to mitigate catastrophic forgetting of general OCR capabilities.

Note: OmniDocBench was rate-limited during data collection and could not be used as the replay source; SynFinTabs was selected as a structurally similar fallback with comparable financial document diversity.

## 3.3 Annotation Schema

All ground truth in FinDocOCR follows a unified output format designed to be parseable by a single VLM decoding pass. The schema extends standard Markdown with minimal, well-scoped HTML inline elements:

| Feature | Ground Truth Format | Example |
|---|---|---|
| Strikethrough | `~~text~~` (Markdown) | `~~Net income: $4.2M~~` |
| Underline | `<u>text</u>` (HTML inline) | `<u>Total assets</u>` |
| Colored text | `<span style="color:NAME;">text</span>` | `<span style="color:red;">Loss from operations</span>` |
| Complex tables | `<table>` with `colspan`/`rowspan` | Full structural HTML |
| Plain text / formulas | Markdown / `$$...$$` LaTeX | Standard OCR output |

The named color vocabulary is restricted to seven colors—`red`, `blue`, `green`, `orange`, `purple`, `yellow`, `gray`—reflecting the semantic palette observed in real financial filings. Black text requires no annotation. This closed vocabulary simplifies evaluation to a classification problem over a fixed label set.

For complex tables, ground truth HTML contains only structural tags and merge attributes; all presentational information (fonts, padding, inline CSS, XBRL namespaced tags) is stripped during post-processing to ensure that ground truth reflects semantic structure rather than rendering style.

## 3.4 Evaluation Set

We construct a held-out evaluation set of **200 pages** (50 per feature) drawn from existing dataset splits. The evaluation set is balanced across features and representative of the full range of formatting complexity in each category.

The evaluation protocol follows a bootstrap-then-annotate design. First, HunyuanOCR (vanilla, no fine-tuning) is run on all 200 pages to generate bootstrap predictions. Two independent annotators then review each page and correct the model output, filling the ground truth field. A third reviewer resolves disagreements. Inter-annotator agreement is measured by Cohen's kappa, with a target threshold of κ ≥ 0.80 before the evaluation set is considered finalized. The bootstrap JSONL is stored at `dataset/eval/eval_bootstrap.jsonl`.

At the time of writing, the evaluation set bootstrap has been generated; human annotation is pending and will be completed before the final training run and evaluation (project plan Phase 3).

## 3.5 Training Set Statistics

The complete FinDocOCR training set is summarized below.

| Feature | Sources | Pairs | ~Tokens | Status |
|---|---|---|---|---|
| Complex tables | FinTabNet.c (2,064) + SynFinTabs test (200) + EDGAR (1,301) | 3,565 | ~4K (current) | Complete; EDGAR pipeline ready to scale |
| Strikethrough | EDGAR amendments (446) + synthetic (5,000) | 5,446 | ~5K | Complete |
| Underline | Synthetic generator | 5,000 | ~5K | Complete |
| Colored text | Synthetic color (5,000) + mixed-feature (5,000) | 10,000 | ~10K | Complete |
| General replay (SynFinTabs) | SynFinTabs train split | 5,000 | ~5K | Complete |
| **Total (excl. replay)** | | **23,011** | **~24K** | |
| **Total (incl. replay)** | | **28,011** | **~29K** | |

The complex table count (3,565) falls short of the 55K target noted in our project roadmap; the EDGAR pipeline is designed to scale to additional filings without code changes. Current counts are sufficient for baseline fine-tuning and ablation experiments. The held-out evaluation set (200 pages) is excluded from all training splits; no training data is drawn from the evaluation JSONL.
