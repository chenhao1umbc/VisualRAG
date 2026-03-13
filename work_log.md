# Work Log — FinDocOCR Paper

## 2026-03-11

### Setup — Complete
- Initialized paper infrastructure: `plan.md`, `to_reviewer.md`, `to_writer.md`, `work_log.md`
- Created directories: `paper/`, `paper/figures/`, `draft/`, `sources/`, `scripts/`
- Reviewed project context: `project_overview_core.md`, `project_plan.md` (367 lines)
- Project status: dataset construction ~90% done (Phases 1.1–1.6 complete, 1.7 annotation pending); training/eval (Phases 2–3) not yet started; paper writing begins now in parallel
- Dataset summary: 3,565 table pairs, 5,446 strikethrough, 5,000 underline, 10,000 color, 5,000 replay; eval set 200 entries bootstrapped
- Key paper contributions identified: (1) FinDocOCR dataset, (2) HunyuanOCR + Dual LoRA, (3) evaluation benchmark
- Research gap confirmed in `project_overview_core.md`: no prior work covers all 4 features together in financial domain
- Cron job started for `to_writer.md` monitoring (ID: see memory)
- Starting Phase 1: Task 1.0 novelty verification

### Reviewer Brief Received — IN-PROGRESS
- Received full initial brief from reviewer via `to_writer.md`
- Paper working title: "FinDocOCR: A Financial Document OCR Dataset and Dual LoRA Benchmark for Strikethrough, Underline, Color, and Complex Table Recognition"
- Tasks 1.0–1.3 all assigned; execution order: 1.0 (blocking) → 1.1 → 1.2 → 1.3
- Beginning Task 1.0 (novelty verification) immediately

### Tasks 1.0–1.3 — COMPLETE (NEEDS REVIEW)
- Searches conducted: strikethrough OCR, underline OCR, colored text OCR, OmniDocBench schema, financial table VLM, Dual LoRA + document OCR
- OmniDocBench annotation schema verified from official GitHub README: 15 block-level, 4 span-level — confirmed NO strikethrough/underline/color
- Dual LoRA equations extracted from arxiv.org/html/2512.03402: full decomposition verified
- 16 papers annotated in `sources/papers_annotated.md` with BibTeX entries
- 3 source files produced: novelty_check.md, dual_lora_notes.md, papers_annotated.md
- Honest caveat: Chinese-language venues not searched; possible gap there

### Reviewer Feedback Applied — Fixes Complete
- Fetched 3 missing citations: Poddar et al. 2021 (ICDAR Workshop), Hase et al. 2003 (IJDAR), Tan et al. 2025 (arXiv:2508.05669)
- Downloaded and OCR'd Dual LoRA PDF (12 pages, GLM-OCR). Equations confirmed: HTML version was correct.
  - ReLU/Sign formulation: $\Delta W = \frac{\alpha}{\sqrt{r_1 r_2}} \text{ReLU}(BA) \odot \text{Sign}(DC)$ — matches PDF exactly
  - Confirmed NLP-only (Section 4): no VLM or document OCR experiments anywhere in paper
  - Added AMD affiliation to dual_lora_notes.md
- Updated novelty_check.md: 3 citations now complete with author lists, volume/page/DOI
- Updated papers_annotated.md: added Poddar 2021, Hase 2003, Tan 2025 to summary table and detailed entries (19 papers total)
- papers_annotated.md was created before reviewer's pre-emptive note — noted in response

### Section 2 (Related Work) — COMPLETE (NEEDS REVIEW)
- Phase 1 confirmed approved by reviewer; Phase 2 cleared
- Wrote `draft/sec2_related_work.md`: 1,467 words, 5 subsections per brief
- All citations drawn from `sources/papers_annotated.md`; no new research needed
- Gap argument closes section 2 with explicit "no existing work" claim supported by named papers

### Section 2 — Reviewer Fixes Applied
- Added DocBank (li2020docbank) and DocLayNet (pfitzmann2022doclaynet) to papers_annotated.md: summary table + detailed entries with BibTeX (21 entries total)
- Fixed HunyuanOCR superlative in §2.1: "current state of the art" → "one of the leading open-weight OCR VLMs"
- GOT-OCR 2.0 parameter count (580M): confirmed correct from arXiv:2409.01704 abstract ("580M parameters") — no change needed

### Section 3 (Dataset) — COMPLETE (NEEDS REVIEW)
- Wrote `draft/sec3_dataset.md`: 1,104 words, 5 subsections per brief
- Statistics table drawn from project_plan.md Phase 1 Dataset Summary
- Honest about annotation-pending status for eval set and EDGAR table scaling gap
- Noted OmniDocBench rate-limit / SynFinTabs fallback as instructed

### Section 3 — Reviewer Fixes Applied
- Fixed SynFinTabs 200-sample contradiction in §3.2: "reserved for baseline evaluation" replaced with reviewer-provided wording clarifying dual-use with no data leakage
- Added ~Tokens column to §3.5 statistics table with values from project plan

### Section 4 (Method) — COMPLETE (NEEDS REVIEW)
- Wrote `draft/sec4_method.md`: 1,199 words, 4 subsections
- ViT/LLM-only decision confirmed from project_plan.md: primary config = LLM-only (ViT frozen); ViT ablation in §5
- All Dual LoRA equations included with full LaTeX; STE explained; DoRA distinction clear
- FinTabNet.c correctly identified as eval-only (not in training split)

### Section 4 — Reviewer Fixes Applied (Resubmission)
- Fix 1: Effective rank corrected in §4.2 — r₁·r₂ (= r² when r₁=r₂=r), not min(d,k)²; strong argument preserved
- Fix 2: Initialization paragraph reframed — now cites Xu et al. §3.3 directly with verbatim quote from PDF OCR; no longer presented as our implementation decision
- Both fixes applied to `draft/sec4_method.md`; resubmission report posted to `to_reviewer.md`
