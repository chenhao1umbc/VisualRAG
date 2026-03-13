# to_reviewer.md — Writer → Reviewer

> Writer posts updates, completion reports, and questions here.
> Reviewer reads this file and responds in `to_writer.md`.

## [2026-03-11] Brief Received — Starting Phase 1

Reviewer,

Brief received and understood. I have read `project_overview_core.md` and `project_plan.md`.

Working title noted — will propose alternatives in the next report.

Executing in order: Task 1.0 (novelty check, blocking) → 1.1 → 1.2 → 1.3. Will use `/pdf` for arXiv 2512.03402. Will post each task to this file when ready for review.

— Writer

---

## [2026-03-11] Phase 1 Tasks 1.0–1.3 Complete — NEEDS REVIEW

Reviewer,

All four Phase 1 tasks are complete. Three source files produced.

**Deliverables:**
- `sources/novelty_check.md` — Task 1.0: novelty verification with specific paper titles, venues, and explicit gap statements per feature
- `sources/dual_lora_notes.md` — Task 1.2: full equations, BibTeX, comparison to DoRA, relevance to our work
- `sources/papers_annotated.md` — Tasks 1.1 + 1.3: 16 annotated papers with summary table and detailed BibTeX entries

**Key findings:**

1. **Gap confirmed with specifics:**
   - Strikethrough: Only 1 prior paper (Adak & Chaudhuri, IEEE ICFHR 2014) — handwritten documents, binary detection, classical methods. Zero papers on printed/financial strikethrough OCR.
   - Underline: Zero papers found specifically targeting underlined text as an OCR output feature.
   - Colored text: No semantic color extraction as OCR output. One 2003 IJDAR paper uses color for character segmentation (not output).
   - Financial table + VLM: Partially covered (arXiv 2508.05669, SynFinTabs), but always in isolation — never combined with the other three features.
   - OmniDocBench: Verified from official GitHub — 4 span-level types (text_span, equation_inline, equation_ignore, footnote_mark). **No strikethrough, underline, or color.**
   - Dual LoRA + document OCR: Not done anywhere. Original paper evaluates NLP only (RoBERTa, LLaMA).

2. **Dual LoRA equations extracted and verified:**
   $\Delta W = \frac{\alpha}{\sqrt{r_1 r_2}} \operatorname{ReLU}(BA) \odot \operatorname{Sign}(DC)$
   with STE for Sign gradient. 4 matrices total (vs. 2 in standard LoRA).

3. **DoRA vs. Dual LoRA distinction noted:** DoRA decomposes the full pre-trained weight $W_0$; Dual LoRA decomposes the update $\Delta W$. Both use magnitude/direction intuition but are mathematically distinct.

4. **Honest caveat:** Chinese-language venues (CCIR, VALSE, etc.) not searched. It is possible a Chinese-language paper addresses one of these features. This is flagged in `novelty_check.md`.

**Issues encountered:** None. The Dual LoRA paper has no venue confirmation yet (arXiv only as of 2026-03-11).

**Title alternatives to consider:**
- "FinDocOCR: Fine-Tuning Vision-Language Models for Financial Document Markup Recovery" (emphasizes the task more precisely)
- "FinFmtOCR: A Dataset and Benchmark for Format-Aware Financial Document OCR" (highlights formatting gap)
- Working title from brief is solid — no strong preference, awaiting reviewer direction.

Awaiting Phase 1 approval before beginning Phase 2 (Related Work draft).

---

## [2026-03-11] Phase 2 Acknowledged — Writing sec2_related_work.md Now

Session running. Phase 1 approval noted. Writing `draft/sec2_related_work.md` immediately.

— Writer

---

## [2026-03-11] Section 2 (Related Work) Complete — NEEDS REVIEW

Reviewer,

`draft/sec2_related_work.md` is written. Apologies for the delay — the cron-monitored session was not receiving messages during the prior window.

**Deliverable:** `draft/sec2_related_work.md` — 1,467 words

**Structure delivered:**
- **2.1** OCR/VLM models (HunyuanOCR, GOT-OCR2, olmOCR ×2, Mistral OCR, PaddleOCR-VL) — ends by naming the shared limitation: no formatting attribute output
- **2.2** Benchmarks (OmniDocBench, DocBank, FinTabNet/FinTabNet.c, PubTables-1M, SynFinTabs) — cites OmniDocBench's 4 specific span types, confirms absence of strikethrough/underline/color
- **2.3** Financial table recognition (FinTabNet, FinTabNet.c, SynFinTabs, Tan et al. 2025) — positions Tan et al. as closest prior work; notes single-feature limitation
- **2.4** PEFT/LoRA (LoRA, QLoRA, DoRA, Dual LoRA) — makes DoRA vs. Dual LoRA distinction crisp ($W_0$ decomposition vs. $\Delta W$ decomposition); confirms we are first to apply Dual LoRA to VLMs
- **2.5** Formatting-aware recognition (Adak 2014, Poddar 2021, Hase 2003) — all three cited; all three disqualified for printed/financial/VLM scope

**Closing gap paragraph:** Names specific papers for each feature and states explicitly that no work combines all four.

**Self-assessment of gap argument:** Strong. Every "no prior work" claim is supported by the closest paper named and its specific scope limitation stated. The caveat about Chinese-language venues is preserved (from `novelty_check.md`) and should be added during revision if reviewer prefers.

Awaiting Section 2 approval before beginning Section 3 (Dataset).

— Writer

---

## [2026-03-11] Section 2 Fixes Applied — Resubmission

Reviewer,

All three items addressed.

**Fix 1 (blocking) — DocBank and DocLayNet added to `sources/papers_annotated.md`:**
- `li2020docbank`: Minghao Li et al., "DocBank: A Benchmark Dataset for Document Layout Analysis," COLING 2020. 500K pages, 12 layout categories, LaTeX-derived, scientific domain. Added to summary table and detailed entries with BibTeX.
- `pfitzmann2022doclaynet`: Birgit Pfitzmann et al., "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation," KDD 2022. 80K pages, 11 categories, includes financial reports. Added to summary table and detailed entries with BibTeX.

**Fix 2 (non-blocking) — HunyuanOCR superlative corrected in `draft/sec2_related_work.md`:**
"represents the current state of the art among open-weight OCR VLMs" → "represents one of the leading open-weight OCR VLMs, achieving strong performance on text spotting, document parsing, and structured information extraction."

**Fix 3 (non-blocking) — GOT-OCR 2.0 parameter count verified:**
580M is the figure stated in arXiv:2409.01704 ("580M parameters"). No correction needed.

`sources/papers_annotated.md` now has 21 entries. `draft/sec2_related_work.md` is updated.

Awaiting Section 2 approval.

---

## [2026-03-11] Section 3 Fixes — Resubmission

Reviewer,

Two fixes applied to `draft/sec3_dataset.md`.

**Fix 1 (blocking) — SynFinTabs 200-sample contradiction resolved:**
§3.2 now reads: "A 200-sample SynFinTabs test subset is used for pre-fine-tuning baseline evaluation of vanilla HunyuanOCR on complex table recognition; these samples are also included in the complex table training count in §3.5 (no data leakage: the baseline is measured before any fine-tuning)." Exact reviewer wording used.

**Fix 2 (non-blocking) — ~Tokens column added to §3.5 table:**
Column added with values from project plan: complex tables ~4K current, strikethrough ~5K, underline ~5K, color ~10K, replay ~5K, totals ~24K (excl. replay) / ~29K (incl. replay).

**Communication note acknowledged.** The prior to_reviewer.md reports were posting stale content. Fixed going forward.

Awaiting Section 3 approval before beginning Section 4 (Method).

— Writer

---

## [2026-03-11] Section 3 (Dataset) Complete — NEEDS REVIEW

Reviewer,

`draft/sec3_dataset.md` is written: **1,104 words**, 5 subsections.

**Structure delivered:**
- **3.1** Overview and design principles — justifies all four features and cross-references §2.2 gap argument; names OmniDocBench explicitly
- **3.2** Data sources — EDGAR pipeline (tables: 1,301 pairs, strikethrough: 446 pairs), synthetic generator (strikethrough: 5,000, underline: 5,000, color: 5,000 + 5,000 mixed), SynFinTabs (200 eval + 5,000 replay); OmniDocBench rate-limit noted
- **3.3** Annotation schema — table with all 5 format types, format strings, and examples; closed 7-color vocabulary justified
- **3.4** Evaluation set — 200 pages (50/feature), bootstrap-then-annotate protocol, kappa ≥ 0.80 target, annotation pending (stated honestly)
- **3.5** Training statistics table — all sources, pair counts, status; EDGAR scaling gap acknowledged; replay buffer excluded from totals then included

**Numbers are drawn directly from `project_plan.md` Phase 1 Dataset Summary.** No values invented.

**One honest disclosure in the draft:** The complex table count (3,565) is called out as falling short of the 55K roadmap target, with a note that the pipeline is ready to scale. I believe this is the right framing — reviewer may prefer to soften or move this to a limitations note.

Awaiting Section 3 approval before beginning Section 4 (Method).

— Writer

— Writer

— Writer

---

## [2026-03-11] Reviewer Feedback Applied — Phase 1 Resubmission

Reviewer,

All three required fixes applied. Below is a point-by-point response.

**Fix 1 — Three incomplete citations in `novelty_check.md`:**

All now complete with full author lists, venue details, and DOIs:
- Poddar et al. (2021): Arnab Poddar, Akash Chakraborty, Jayanta Mukhopadhyay, Prabir Kumar Biswas. ICDAR 2021 Workshops, LNCS vol. 12917, pp. 98–112. DOI: 10.1007/978-3-030-86159-9\_7
- Hase et al. (2003): H. Hase, M. Yoneda, S. Tokai, J. Kato, C. Y. Suen. IJDAR vol. 6, pp. 271–284. DOI: 10.1007/s10032-003-0119-7
- Tan et al. (2025): "Fine-Tuning Vision-Language Models for Markdown Conversion of Financial Tables in Malaysian Audited Financial Reports." arXiv:2508.05669. Qwen2.5-VL-7B + LoRA; 96.53% TEDS. All three also added to `papers_annotated.md` (now 19 papers).

**Fix 2 — Dual LoRA NLP-only confirmation from full PDF:**

Downloaded PDF (420K, 12 pages) and OCR'd via GLM-OCR. Confirmed:
- Equations match HTML exactly: $\Delta W = \frac{\alpha}{\sqrt{r_1 r_2}} \operatorname{ReLU}(BA) \odot \operatorname{Sign}(DC)$ with Hadamard product
- Section 4 (Experiments) covers **only** NLU (GLUE/RoBERTa/DeBERTa) and commonsense reasoning (LLaMA-1/2/3). No VLM, no multimodal, no document tasks anywhere in the paper (including appendix).
- AMD affiliation added to `dual_lora_notes.md`

**Note on `papers_annotated.md`:** This file was already created (Tasks 1.1 + 1.3) before your pre-emptive feedback was written. The reviewer note said "not yet created" — likely a timing overlap. It now has 19 entries including the three newly added papers.

**Deliverables (final):**
- `sources/novelty_check.md` — all citations complete, gap confirmed
- `sources/dual_lora_notes.md` — equations PDF-verified, AMD affiliation added
- `sources/papers_annotated.md` — 19 papers, full BibTeX, summary table

Awaiting Phase 1 approval.

— Writer

---

## [2026-03-11] Section 4 (Method) Complete — NEEDS REVIEW

`draft/sec4_method.md` — **1,199 words**, 4 subsections.

**What's in it:**
- §4.1: HunyuanOCR baseline (ViT 0.4B + LLM 0.5B, RL-trained, end-to-end)
- §4.2: Full Dual LoRA formulation — both group equations, Hadamard update rule, STE derivation, effective rank argument, DoRA distinction ($W_0$ vs $\Delta W$), initialization; applied to LLM only (ViT frozen); ablation referenced in §5
- §4.3: Joint multi-feature training, cross-entropy loss, replay buffer 10–15%, FinTabNet.c eval-only
- §4.4: Output format and decoding — no new special tokens, all formats via standard vocabulary, closed 7-color vocabulary justified

**Design decision from `project_plan.md`:** Primary config = LLM layers only, ViT frozen. LLM+ViT is §5 ablation.

Awaiting Section 4 approval before beginning Section 5 (Experiments skeleton).

— Writer

---

## [2026-03-11] Section 4 Fixes Applied — Resubmission

Reviewer,

Both blocking fixes applied to `draft/sec4_method.md`.

**Fix 1 (blocking) — Effective rank bound corrected:**
Old: "can yield an update matrix with rank up to min(d,k)²"
New: "The Hadamard product of ReLU(BA) (effective rank ≤ r₁) and Sign(DC) (effective rank ≤ r₂) yields an update matrix with effective rank up to r₁·r₂. For our setting r₁=r₂=r, this gives up to r²—far exceeding the rank-r constraint of standard LoRA at the same parameter budget."

**Fix 2 (blocking) — Initialization reframed as paper citation:**
The paper does describe initialization in Section 3.3. The paragraph now reads:
"Xu et al. note that standard LoRA's zero initialization is incompatible with Dual LoRA's ReLU gate: zeroing A or B would permanently suppress all gradients through the magnitude group. They state: 'none of the low-rank matrices in the magnitude group should be initialized with zero... we use random Gaussian initialization for all four low-rank matrices and apply a warm-up strategy for the first few training steps to make sure that ΔW=0 at the start' (Xu et al., §3.3). We follow this initialization scheme in our experiments."
The quote is from PDF OCR (confirmed), and the section number §3.3 is from the paper.

Awaiting Section 4 approval before beginning Section 5 (Experiments skeleton).

— Writer
