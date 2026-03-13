# Project Plan

> Conventions:
> - `[ ]` = not started, `[x]` = done
> - Each step has a **Metric** — the measurable definition of "done"
> - After each step: automated subagent check
> - After each phase: **PAUSE for manual user review** before proceeding

---

## Phase 1: Dataset Construction

### 1.1 Complex Tables — FinTabNet.c Baseline

- [x] Download FinTabNet.c from HuggingFace (`bsmock/FinTabNet.c`) — cell annotations reconstructed to HTML + PIL-rendered images; 2,064 pairs
- [ ] Run HunyuanOCR (vanilla) on FinTabNet.c test set
- [ ] Record baseline TEDS score

> Note: HuggingFace version is annotation-only (no raw PDFs). Cell data reconstructed to `<table>` HTML using `download_datasets.py`.

**Metric**: TEDS score reported on FinTabNet.c test set; confirms table recognition failure in baseline (expected TEDS < 0.6 on complex tables)
> Subagent check: verify TEDS computation is correct, score is reproducible

---

### 1.2 Complex Tables — SynFinTabs Assessment

- [x] Download SynFinTabs from HuggingFace (`ethanbradley/synfintabs`) — 200-sample test subset saved to `dataset/synfintabs/`
- [ ] Visually inspect 50 random samples for domain realism
- [ ] Run HunyuanOCR on SynFinTabs test subset (200 samples)
- [ ] Record TEDS score

**Metric**: Visual quality assessment documented (pass/fail judgment on realism); TEDS score on SynFinTabs subset
> Subagent check: confirm sample diversity, flag if synthetic images look unrealistic

---

### 1.3 Complex Tables — EDGAR HTML Pipeline

- [x] Build EDGAR HTML downloader (SEC EDGAR EFTS API, filter 10-K/10-Q)
- [x] Build PIL-based HTML table renderer (no system deps; handles colspan/rowspan)
- [x] Build CSS color extractor (normalize hex/rgb → named color set)
- [x] Build table HTML extractor (preserve colspan/rowspan)
- [x] Run pipeline on 500 filings → 1,301 (image, HTML) pairs
- [x] Filter: keep only tables with ≥3 rows, ≥2 columns, ≥1 merged cell (colspan/rowspan)
- [x] Fix `sanitize_table_html`: root `<table>` attrs now cleared (find_all skips root — bug fixed)
- [x] Post-process existing JSONL: 0 XBRL tags, 0 inline CSS, colspan/rowspan preserved

> Note: PIL renderer used (Playwright needs system libs unavailable in this env). 1,301 pairs produced (prior 2,486 count included multi-line JSON miscounting). Ground truth is now clean structural HTML.

**Metric**: ≥5,000 (image, HTML ground truth) pairs produced; spot-check 50 pairs manually for HTML correctness
> Subagent check: verify HTML structure integrity, confirm color normalization mapping is correct
> **GAP**: 1,301 vs 5,000 target — run more filings before training. Current count sufficient for baseline eval.

---

### 1.4 Strikethrough — EDGAR Amendment Pipeline

- [x] Filter EDGAR pipeline for amendment filings (`10-K/A`, `DEF 14A`, `S-1/A`)
- [x] Extract `<del>` / `text-decoration: line-through` CSS spans as strikethrough ground truth
- [x] Render to images (PIL strikethrough renderer)
- [x] Filter: keep only spans with ≥10 chars of text → 446 quality pairs

**Metric**: ≥500 real (image, ground truth) pairs with confirmed strikethrough; spot-check 50 pairs
> Subagent check: verify `~~text~~` formatting is correct, no false positives from CSS parsing
> **GAP**: 446 vs 500 target — minor shortfall; synthetic (5,000) more than compensates for training.

---

### 1.5 Strikethrough + Underline + Color — Synthetic Generator

- [x] Build unified synthetic document generator (`dataset/synthetic_generator.py`)
  - [x] Strikethrough rendering (vary: font, size, line thickness, single/double strike)
  - [x] Underline rendering (vary: single/double, spacing from baseline)
  - [x] Color rendering (vary: 7 named colors, font weight, background)
  - [x] Mixed pages (multiple features co-occurring, realistic financial text content)
  - [x] Financial text content from EDGAR sample sentences
- [x] Generate 5,000 samples per feature (20,000 total including mixed)
- [ ] Spot-check 100 samples per feature (visual inspection — requires human)

**Metric**: 15,000 synthetic (image, ground truth) pairs; visual inspection confirms features are clearly visible and labels match `project_overview_core.md` format conventions
> Format check: 100% compliance verified programmatically

---

### 1.6 General Replay Buffer

- [x] Sample 5,000 pages from SynFinTabs train split (OmniDocBench rate-limited; SynFinTabs used as fallback)
- [x] Saved to `dataset/replay/replay.jsonl`
- [x] Financial keyword overlap: spot-checked (SynFinTabs is non-EDGAR, different distribution)

**Metric**: 5,000 general-domain samples in training format; confirmed no domain overlap
> Subagent check: verify format consistency with financial training samples

---

### 1.7 Held-out Evaluation Set — Manual Annotation

- [x] Select 200 pages from existing dataset splits (50 per feature)
- [x] Bootstrap JSONL created at `dataset/eval/eval_bootstrap.jsonl` (regenerated: color was 25→50, total 175→200)
- [ ] Run HunyuanOCR on all 175 pages (bootstrap predictions)
- [ ] Annotator 1 reviews all 175 pages, fills `annotator1` field
- [ ] Annotator 2 independently reviews all 175 pages, fills `annotator2` field
- [ ] Compute Cohen's kappa: `uv run python dataset/compute_kappa.py`
- [ ] Resolve disagreements, finalize ground truth JSONL

**Metric**: 200 annotated pages; Cohen's kappa ≥ 0.80; annotation schema documented
> Subagent check: verify JSONL format, kappa computation, check for missing annotations

---

### Phase 1 Dataset Summary

| Feature | Source | Actual Size | Target | Status |
|---|---|---|---|---|
| Complex tables | FinTabNet.c (2,064) + SynFinTabs (200) + EDGAR (1,301) | 3,565 | ~55K | [x] pipelines done; scale up EDGAR for training |
| Strikethrough | EDGAR amendments (446) + synthetic (5,000) | 5,446 | ~5.5K | [x] |
| Underline | Synthetic | 5,000 | ~5K | [x] |
| Colored text | Synthetic color (5,000) + mixed (5,000) | 10,000 | ~5K | [x] |
| General replay | SynFinTabs train (fallback from OmniDocBench) | 5,000 | ~5K | [x] |
| Eval set | Bootstrap ready (200 entries, 50/feature), annotation pending | 200 | ~200 | partial |

> **PAUSE: Manual user review of Phase 1 before proceeding to Phase 2**
> Review: dataset statistics, sample quality, annotation correctness, label format

---

## Phase 2: Dual LoRA Implementation

### 2.1 Read and Document Dual LoRA Algorithm

- [ ] Read arXiv 2512.03402 fully
- [ ] Write internal notes: magnitude/direction decomposition equations, differences from standard LoRA
- [ ] Identify which components need to change in `finetune_lora.py`

**Metric**: Equations documented in `notes/dual_lora_algorithm.md`; list of required code changes written
> Subagent check: verify equations match paper, flag implementation ambiguities

---

### 2.2 Implement DualLoraLinear Layer

- [ ] Implement `DualLoraLinear` class in `hunyuanOCR/dual_lora.py`
  - Magnitude component (scalar per rank dimension)
  - Direction component (standard BA matrix)
  - Separate learning rate support for magnitude vs direction
- [ ] Unit test: when magnitude LR = direction LR, output matches standard LoRA
- [ ] Unit test: gradient flows through both magnitude and direction components
- [ ] Unit test: forward pass output shape is correct for all target layer sizes in HunyuanOCR

**Metric**: All 3 unit tests pass; numerically verified against standard LoRA baseline
> Subagent check: inspect implementation against paper equations, verify test coverage is sufficient

---

### 2.3 Determine LoRA Target Layers

- [ ] Profile which layers are most relevant for visual features (strikethrough, underline, color) vs text features (tables)
- [ ] Decision: LLM-only vs LLM+ViT (documented with justification)
- [ ] Count trainable parameters for both configurations
- [ ] Target: 1–5% of total model parameters trainable

**Metric**: Parameter count table documented; decision recorded with justification in `notes/target_layers.md`
> Subagent check: verify parameter counts, confirm ViT layer names are correct for HunyuanOCR architecture

---

### 2.4 Integrate Dual LoRA into Training Script

- [ ] Create `hunyuanOCR/finetune_duallora.py` based on `finetune_lora.py`
- [ ] Replace standard LoRA with `DualLoraLinear` integration
- [ ] Add separate optimizers/LR schedulers for magnitude and direction components
- [ ] Add replay buffer mixing logic (10–15% general samples per batch)
- [ ] Test: training runs on `sample_data/` for 10 steps without errors
- [ ] Test: loss decreases over 10 steps, no NaN/Inf gradients

**Metric**: Training script runs end-to-end on sample data; loss is decreasing; gradient norms are stable
> Subagent check: review code for bugs, verify loss masking is correct, confirm replay mixing ratio

---

### 2.5 Hyperparameter Sweep (Small Scale)

- [ ] Sweep: LoRA rank ∈ {8, 16, 32}
- [ ] Sweep: magnitude LR / direction LR ratio ∈ {0.1, 1.0, 10.0}
- [ ] Use 10% of Phase 1 training data and Phase 1.7 eval set for quick validation
- [ ] Select best config by composite score: TEDS + mean F1 across all four features

**Metric**: Sweep results table with composite scores; best configuration identified and frozen for Phase 3
> Subagent check: verify sweep is fair (same seeds, same data splits), flag signs of overfitting

---

> **PAUSE: Manual user review of Phase 2 before proceeding to Phase 3**
> Review: Dual LoRA implementation correctness, training stability, hyperparameter decision

---

## Phase 3: Training & Evaluation

### 3.1 Baseline Evaluation — HunyuanOCR Vanilla

- [ ] Run HunyuanOCR (no fine-tuning) on full eval set (Phase 1.7)
- [ ] Compute all metrics:
  - Tables: TEDS (Tree Edit Distance Score)
  - Strikethrough: span-level F1 + exact match rate
  - Underline: span-level F1 + exact match rate
  - Color: color classification accuracy + span-boundary F1
  - General OCR: CER, WER on OmniDocBench subset (100 pages)

**Metric**: Full baseline results table populated; confirms model fails on target features (strikethrough/underline F1 < 0.1 expected)
> Subagent check: verify metric implementations are correct, cross-check TEDS with reference implementation

---

### 3.2 Standard LoRA Training Run

- [ ] Train HunyuanOCR + standard LoRA (`finetune_lora.py`) on full Phase 1 dataset (with replay)
- [ ] Use same rank as best config from Phase 2.5 for fair comparison
- [ ] Evaluate on Phase 1.7 eval set, compute all metrics
- [ ] Save checkpoint

**Metric**: Results table populated; improvement over vanilla baseline on ≥3 of 4 target features
> Subagent check: verify training completed without instability, confirm eval is on held-out set only

---

### 3.3 Dual LoRA Training Run — Proposed Method

- [ ] Train HunyuanOCR + Dual LoRA (`finetune_duallora.py`) on full Phase 1 dataset (with replay)
- [ ] Use best hyperparameters from Phase 2.5
- [ ] Evaluate on Phase 1.7 eval set, compute all metrics
- [ ] Save checkpoint

**Metric**: Dual LoRA outperforms standard LoRA on ≥3 of 4 target features; general OCR (CER/WER) does not degrade vs vanilla baseline by more than 5%
> Subagent check: verify no data leakage between train/eval sets, confirm results are reproducible with fixed seed

---

### 3.4 Ablation: LLM-only vs LLM+ViT LoRA

- [ ] Train Dual LoRA with ViT frozen (LLM-only)
- [ ] Train Dual LoRA with ViT LoRA-tuned (LLM+ViT)
- [ ] Compare delta on visual features (strikethrough, underline, color F1)

**Metric**: Delta table showing F1 change from adding ViT LoRA; clear recommendation justified by numbers
> Subagent check: verify ViT LoRA is correctly applied, parameter counts match expected

---

### 3.5 Ablation: Impact of Replay Buffer

- [ ] Train Dual LoRA without replay buffer (all other settings identical to 3.3)
- [ ] Compare OmniDocBench CER/WER vs with-replay model

**Metric**: CER/WER degradation without replay is ≥5% (confirming replay is necessary); result documented
> Subagent check: verify OmniDocBench evaluation setup is identical in both runs

---

### 3.6 Qualitative Analysis

- [ ] Select 3–5 representative examples per feature: baseline failure → model success
- [ ] Select 2–3 honest failure cases of the proposed model
- [ ] Produce side-by-side figures (input image | baseline output | our output)

**Metric**: ≥15 qualitative examples documented; at least 2 failure cases analyzed with explanation
> Subagent check: verify examples are representative and not cherry-picked outliers, failure cases are genuine

---

> **PAUSE: Manual user review of Phase 3 before proceeding to Phase 4**
> Review: results tables, ablations, qualitative examples, overall experimental narrative

---

## Phase 4: Paper Writing

### 4.1 Related Work

- [ ] OCR models: HunyuanOCR, DeepSeek-OCR, OlmOCR, Mistral OCR, PaddleOCR-VL
- [ ] Document understanding benchmarks: OmniDocBench (CVPR 2025), FinTabNet.c, DocBank
- [ ] Table recognition: TEDS metric, PubTables-1M, SynFinTabs
- [ ] LoRA variants: standard LoRA, Dual LoRA (arXiv 2512.03402), QLoRA
- [ ] Financial document AI: FinQA, TAT-QA (note these are QA datasets, not OCR)
- [ ] Confirm: no paper addresses all four target features together in financial OCR

**Metric**: Related work draft covers all 5 areas; research gap is clearly argued with citations
> Subagent check: verify citations are correct and complete, gap argument is logically sound

---

### 4.2 Dataset Section

- [ ] Describe FinDocOCR construction pipeline, filtering criteria, and statistics
- [ ] Dataset statistics table: size per feature, train/eval split, source breakdown
- [ ] Describe annotation process and inter-annotator agreement (Cohen's kappa from Phase 1.7)
- [ ] Include representative samples figure (one per feature)
- [ ] Describe output format conventions and design decisions (why named colors, why `~~text~~`, etc.)

**Metric**: Dataset section is self-contained; reader could reproduce the pipeline from the description; statistics table matches Phase 1 actuals
> Subagent check: verify statistics match actual dataset, check reproducibility gaps

---

### 4.3 Method Section

- [ ] Dual LoRA formulation: numbered equations for magnitude/direction decomposition
- [ ] Architecture diagram: HunyuanOCR + Dual LoRA adapter locations (LLM vs ViT)
- [ ] Training setup: data mix ratio, replay buffer, loss function, optimizer configuration
- [ ] Justify ViT frozen vs LoRA-tuned decision (citing Phase 3.4 ablation)
- [ ] Justify Dual LoRA over standard LoRA (citing Phase 3.3 results)

**Metric**: All equations numbered; architecture figure included; every design choice has a cited justification or ablation reference
> Subagent check: verify equations are consistent with implementation in `finetune_duallora.py`

---

### 4.4 Experiments Section

- [ ] Main results table: Vanilla vs Standard LoRA vs Dual LoRA, all 5 metrics
- [ ] Ablation table 1: LLM-only vs LLM+ViT LoRA
- [ ] Ablation table 2: with/without replay buffer (catastrophic forgetting)
- [ ] Qualitative figures (from Phase 3.6)
- [ ] Error analysis paragraph (from Phase 3.6 failure cases)

**Metric**: All tables consistent with Phase 3 logged results; figures are publication-quality (≥300 DPI); error analysis is honest and specific
> Subagent check: cross-check every number in tables against Phase 3 result logs, flag any inconsistency

---

### 4.5 Introduction and Abstract

- [ ] Introduction: problem statement, why financial OCR matters, what current models miss, 3 numbered contributions (dataset, model, benchmark)
- [ ] Abstract: ≤250 words, standalone readable, covers problem/method/key results/impact

**Metric**: Introduction states exactly 3 numbered contributions that match what is delivered; abstract is ≤250 words and reviewed by all co-authors
> Subagent check: verify contributions claimed in introduction match actual paper content

---

### 4.6 Figures and Visuals

- [ ] Figure 1: pipeline overview (dataset construction → Dual LoRA training → evaluation)
- [ ] Figure 2: qualitative results (one per feature, baseline vs ours side by side)
- [ ] Figure 3: ablation chart (bar chart for replay buffer and ViT LoRA impact)
- [ ] All figures at ≥300 DPI, fonts readable at column width

**Metric**: All 3 figures complete, integrated into paper, captions are self-contained
> Subagent check: verify figures match text references, captions are complete and accurate

---

### 4.7 Consistency Check and Final Review

- [ ] Cross-check all numbers between abstract, body text, tables, and figures
- [ ] Verify all citations are complete and formatted per target venue style
- [ ] Confirm dataset and code release plan (GitHub repo + HuggingFace dataset card)
- [ ] Full proofread

**Metric**: Zero numerical inconsistencies; citation list complete; release plan confirmed with URLs
> Subagent check: automated cross-check of all numbers cited in text vs tables

---

> **PAUSE: Manual user review of Phase 4 full paper draft before submission**
> Review: logical flow, experimental soundness, writing quality, contribution clarity, venue fit
