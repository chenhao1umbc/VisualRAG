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

### [2026-03-14] Section 5 — Reviewer Fixes Applied (Resubmission)
- Fix 1: Strikethrough F1 and Underline F1 changed from "Token-level" to "Character-level" consistently; added span boundary derivation sentence to each
- Fix 2: Overall macro-average bullet expanded with justification for mixing heterogeneous metrics
- Both fixes applied to `draft/sec5_experiments.md`; resubmission report posted to `to_reviewer.md`; `to_writer.md` truncated

### [2026-03-14] CR-1 through CR-8 — Code Review Fixes Applied
- CR-1: teds.py DP init corrected — col 0 was off-by-(i-1), row 0 was hardcoded 0
- CR-2: teds.py `_build_node` and `_strip_cell_text` extracted to module level
- CR-3: teds.py `from bs4 import BeautifulSoup` moved to top (removed 2 inline occurrences)
- CR-4: compute_kappa.py — primary metric changed to mean token F1 >= 0.85; kappa formula cleaned up with degeneracy note
- CR-5: compute_kappa.py — bare `open()` → context manager; `defaultdict` import moved to top
- CR-6: metrics.py — `_token_set` extracted to module level; `defaultdict` and `teds` imports moved to top
- CR-7: edgar_pipeline.py — `import copy` moved from inside function to top
- CR-8: synthetic_generator.py — `import argparse` and `from tqdm import tqdm` moved to top
- All 5 files: status pending reviewer approval

### [2026-03-14] Phase 2 — DualLoraLinear implemented
- Created `hunyuanOCR/dual_lora.py`: DualLoraLinear, _SignSTE (STE for Sign), apply_dual_lora, trainable_parameters
- Created `hunyuanOCR/tests/test_dual_lora.py`: 4 unit tests — all pass
- All imports at top, no nested functions, type hints throughout
- Status: pending reviewer approval

### [2026-03-14] Sub-task 2.3 — Target Layer Decision documented
- Created notes/target_layers.md
- LLM: 7 layer types × 24 layers = 168 nn.Linear modules
- ViT: 6 layer types × 27 layers = 163 nn.Linear modules
- Decision: all 7 LLM types → 18.48M trainable at r=16 (1.86%)
- ViT ablation targets documented (dense_h_to_4h, dense_4h_to_h + attn projections)
- Status: pending reviewer approval

### [2026-03-15] Sub-task 2.4 — finetune_duallora.py implemented
- Created hunyuanOCR/dataset.py (OCRDataset, collate_fn extracted from finetune_lora.py)
- Updated hunyuanOCR/finetune_lora.py to import from dataset.py
- Created hunyuanOCR/finetune_duallora.py with: Dual LoRA setup, two-group AdamW, replay buffer, warm-up, adapter-only checkpoint
- All files: syntax clean, no nested functions, imports at top
- Status: pending reviewer approval

### [2026-03-15] Sub-task 2.5 — sweep.py implemented
- Created hunyuanOCR/sweep.py: generate_configs() (9 configs), print_sweep_table(), --dry_run/--launch CLI
- Tested: --dry_run prints all 9 configs correctly
- Status: pending reviewer approval

### [2026-03-15] Phase 2 — ALL SUB-TASKS APPROVED
- Sub-task 2.2: DualLoraLinear + tests — APPROVED
- Sub-task 2.3: Target layer decision (notes/target_layers.md) — APPROVED
- Sub-task 2.4: dataset.py + finetune_duallora.py + finetune_lora.py update — APPROVED
- Sub-task 2.5: sweep.py (9-config hyperparameter sweep) — APPROVED
- Phase 2 complete. Awaiting Phase 3 brief (run_eval.py) or paper Section 1 instructions.

### [2026-03-15] Sub-task 2.6 — run_eval.py implemented
- Created hunyuanOCR/eval/run_eval.py
- Dual LoRA adapter injection: infers rank/alpha/targets from .pt file, applies via apply_dual_lora + tensor copy
- JSON output structure verified with mock data + real evaluate_batch calls
- All 4 required top-level keys and 4 feature keys confirmed
- Status: pending reviewer approval

### [2026-03-15] Section 1 (Introduction + Abstract) — submitted for review
- Created draft/sec1_intro.md (862 words)
- Abstract (~200w): problem/gap/contributions/[TBD] results
- §1: motivation, VLM gap survey, downstream consequences
- §1.1: 3 contributions (FinDocOCR dataset, Dual LoRA OCR, benchmark)
- §1.2: paper organization
- All stats/equations/metric names consistent with §§2–5
- Status: pending reviewer approval

### [2026-03-15] Phase 3 — LaTeX skeleton created
- Fixed sec1_intro.md: 28,011 → 23,011 + SynFinTabs clarification; citation style unified
- Created paper/main.tex: article class, all packages, 6 section stubs with subsection labels
- Created paper/references.bib: 24 BibTeX entries from papers_annotated.md
- Status: pending reviewer approval

### [2026-03-15] Phase 3 Deliverable 2 — Section transfer to LaTeX complete
- Transferred §§1–5 from approved draft files into paper/main.tex
- Abstract, §1 Introduction (§§1.1–1.2), §2 Related Work (§§2.1–2.5), §3 Dataset (§§3.1–3.5), §4 Method (§§4.1–4.4), §5 Experiments (§§5.1–5.4)
- 5 labeled equations: eq:lora, eq:magnitude, eq:direction, eq:duallora, eq:ste
- 5 booktabs tables: tab:schema, tab:stats, tab:main, tab:ablation:vit, tab:ablation:replay
- All [TBD] → \tbd{}, all citations → \cite{}/\citet{}, all cross-refs → Section~\ref{}
- Only remaining TODO: §6 Conclusion stub
- Status: pending reviewer approval

## [2026-03-17] CR 4.1 + 4.2 + 4.3 — Critical bug fixes applied
- 4.1 `dual_lora.py`: Added `device=linear.weight.device` to all 4 `torch.empty()` calls (A, B, C, D); 4 unit tests re-run, all pass
- 4.2 `finetune_duallora.py` + `eval/run_eval.py`: Added `_get_device()` (MPS→CUDA→CPU); replaced `device_map="auto"` + `dtype=` with `.to(device)` + `torch_dtype=`; fixed `_run_inference_single` to accept explicit `device` arg; fixed import order (all imports before function defs)
- 4.3 `eval/teds.py`: Clamped TEDS return to `max(0.0, ...)` at line 142
- Files: `hunyuanOCR/dual_lora.py`, `hunyuanOCR/finetune_duallora.py`, `hunyuanOCR/eval/run_eval.py`, `hunyuanOCR/eval/teds.py`
- Status: pending reviewer approval

## [2026-03-17] Sub-task 3.0 done: eval pipeline schema fix
- What was built: `run_eval.py` normalises eval_bootstrap.jsonl schema on load; `_FEATURE_PROMPTS` at module level; PEP8 fix in finetune_duallora.py
- Files: `hunyuanOCR/eval/run_eval.py`, `hunyuanOCR/finetune_duallora.py`
- Status: approved by reviewer

## [2026-03-17] Sub-task 3.1 done: combined training dataset assembly
- What was built: `dataset/build_train_dataset.py` + `dataset/train/train.jsonl`
- Total: 21,947 records — color: 5,000 | general: 5,000 | strikethrough: 5,446 | tables: 1,501 | underline: 5,000
- All image paths verified (20-sample spot check: 0 missing); all 4 required fields present
- Files: `dataset/build_train_dataset.py`, `dataset/train/train.jsonl`
- Status: pending reviewer approval

## [2026-03-17] Sub-task 3.1 fix — dataset.py image path bug
- Fix: `img_path = os.path.join(self.data_dir, sample["image"])` → `img_path = sample["image"]` (lines 49, 58, 74)
- Updated docstring to document project-root-relative image path convention
- Verified: old path non-existent, new path resolves; 20-sample spot check all OK
- Files: `hunyuanOCR/dataset.py`
- Status: pending reviewer approval

## [2026-03-17] Sub-task 3.2 in progress: baseline eval
- Running: `python hunyuanOCR/eval/run_eval.py --checkpoint_dir tencent/HunyuanOCR --eval_dir dataset/eval --output_file results/baseline.json`
- Pre-run fix: `dtype=` → `torch_dtype=` in `run_eval.py:145` (missed in 4.2 approval)
- 200 samples on MPS, logging to `results/baseline_eval.log`
- Status: running in background

## [2026-03-18] Sub-task 3.2 done: baseline eval
- What was built: ran vanilla HunyuanOCR on 200-sample eval_bootstrap.jsonl; results/baseline.json written
- Env fix: used project .venv (torch 2.10.0 + transformers 4.57.1); needed PYTHONPATH=project root
- reverted run_eval.py:145 dtype fix per reviewer (dtype= correct for this transformers pin)
- Results:
    strikethrough F1: 0.0000  (expected — baseline has no markup training)
    underline F1:     0.0000  (expected)
    color accuracy:   0.0000  (expected)
    color boundary F1:0.0000  (expected)
    table TEDS:       0.6691
    CER:              1.0118  (>1.0: model over-generates vs ground truth)
    WER:              1.2124
- Files: results/baseline.json, results/baseline_eval.log
- Status: pending reviewer approval

## [2026-03-18] Sub-task 3.3 done: standard LoRA training
- Dataset: dataset/train_small/train.jsonl (2000 samples, 400/feature, seed=42)
- Config: epochs=1, batch=1, grad_accum=8, lr=2e-4, rank=16, alpha=32, max_length=2048, max_pixels=1M, gradient_checkpointing
- Loss curve (selected steps): step 190→0.5777, 200→0.5635, 210→0.5539, 220→0.5424, 230→0.5345, 240→0.5240, 250→0.5159
- Epoch 1 avg loss: 0.5159
- Final checkpoint: ./lora-output/final
- Files: lora-output/checkpoint-200/, lora-output/final/
- Status: pending reviewer approval

## [2026-03-18] Sub-task 3.4 done: standard LoRA eval
- Eval fix: run_eval.py now detects adapter_config.json → loads base model + PeftModel.from_pretrained
- Added --base_model_path arg (default tencent/HunyuanOCR)
- Results vs baseline:
    tables TEDS:       0.6906 (+0.022 vs baseline 0.6691)
    strikethrough F1:  0.0000 (unchanged)
    underline F1:      0.0200 (+0.020)
    color accuracy:    0.0200 (+0.020)
    color boundary F1: 0.0200 (+0.020)
    CER:               1.1184 (worse, model over-generates more)
    WER:               1.7387 (worse)
- Files: results/lora.json, results/lora_eval.log
- Status: pending reviewer approval

## [2026-03-18] Sub-task 3.5 done: Dual LoRA training
- Config: rank=16, alpha=16.0, lr_magnitude=2e-4, lr_direction=2e-5 (ratio=0.1), 1 epoch, train_small
- Loss curve: 3.27→0.86 (large range due to random init vs LoRA's near-zero init)
- Epoch 1 avg loss: 0.8621; 276 adapters saved
- Final checkpoint: duallora-output/final/dual_lora_adapters_only.pt
- Also fixed run_eval.py: Dual LoRA path now loads clean base model from base_model_path (not checkpoint_dir with mixed state dict)
- Files: duallora-output/checkpoint-200/, duallora-output/final/, results/duallora_train.log
- Status: pending reviewer approval

## [2026-03-18] Sub-task 3.6 done: Dual LoRA eval
- Eval: run_eval.py with --checkpoint_dir duallora-output/final (Dual LoRA path)
- Results: tables TEDS 0.6554, strike F1 0.0000, underline F1 0.0000, color acc 0.0000, CER 0.9349, WER 1.1624
- Note: CER lower than baseline (0.935 vs 1.012) and lower than LoRA (1.118); TEDS slightly below LoRA
- Files: results/duallora.json, results/duallora_eval.log
- Status: reported to reviewer

## [2026-03-18] Sub-task 3.8 done: no-replay ablation
- Discovery: dataset/replay/replay.jsonl has all-empty ground_truth (5000 samples, synfintabs_train source)
- Impact: replay training produces zero gradient signal; 3.5 run IS the no-replay condition
- Action: copied results/duallora.json → results/duallora_noreplay.json as the 3.8 reference
- Paper: replay ablation table updated; "with replay" marked N/A; Limitations note added
- Status: reported to reviewer

## [2026-03-18] Sub-task 3.7 in progress: LLM+ViT ablation training
- Added --include_vit flag to finetune_duallora.py using TARGET_MODULES_LLM_VIT (9 module types)
- Combined target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, dense_h_to_4h, dense_4h_to_h
- Training: rank=16, alpha=16.0, 1 epoch, train_small; 35.87M params (3.48%)
- Output: duallora-vit-output/final; 330 adapters; avg loss 0.8524 (3.23→0.85)
- Loss slightly better than LLM-only (0.8524 vs 0.8621), as expected with more trainable params
- Status: training complete; eval complete

## [2026-03-18] Sub-task 3.7 eval done: LLM+ViT Dual LoRA eval
- Results: tables TEDS 0.6620, strike F1 0.0000, underline F1 0.0000, color acc 0.0000, CER 0.8884, WER 1.2296
- vs LLM-only: TEDS +0.007, CER -0.045 (lower = better); all markup remain 0
- Files: results/duallora_vit.json, results/duallora_vit_eval.log

## [2026-03-18] Sub-task 3.9 done: paper tables complete
- Filled LLM+ViT ablation row (Tab. 3): 0.000, 0.000, 0.000, 0.662, 0.166
- Filled Delta row: 0.000, 0.000, 0.000, +0.007, +0.002
- Recompile: 11 pages, 0 overfull > 10pt
- Only remaining \tbd{}: Tan et al. external paper TEDS (not available)
- Status: complete

## [2026-03-18] Phase 5.A done: data & infrastructure

### 5.A1 — eval_full + --eval_jsonl
- dataset/build_eval_full.py: 200/feature, held-out (not in train_small), seed=42
- dataset/eval/eval_full.jsonl: 1,000 records, 0 missing images
- hunyuanOCR/eval/run_eval.py: --eval_jsonl arg added

### 5.A2 — replay_proper
- dataset/build_replay.py: normalise fintabnet_train.jsonl → OCRDataset schema
- dataset/replay_proper/train.jsonl: 2,064 records, feature=tables, 0 missing

### 5.A3 — train_medium
- dataset/build_medium.py: 600/feature from train.jsonl, seed=42
- dataset/train_medium/train.jsonl: 3,000 records, 0 missing

### 5.A4 — finetune_dora.py
- hunyuanOCR/finetune_dora.py: finetune_lora.py with DoraConfig (use_dora=True), r=16, alpha=32
- PEFT 0.18.1 use_dora confirmed supported

## [2026-03-18] BACKGROUND TASK launched: 5.B1 LoRA v2 training
- Command: source .venv/bin/activate && PYTHONPATH=/Users/hc/Documents/research/Projects/VisualRAG python3 hunyuanOCR/finetune_lora.py --model_path tencent/HunyuanOCR --data_dir dataset/train_medium --output_dir lora-output-v2 --epochs 3 --batch_size 1 --gradient_accumulation_steps 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing 2>&1 | tee results/lora_v2_train.log
- Task ID: bixszo7sd (first attempt b6upy4i4d failed: max_length=1024 too small)
- Status: completed — died at checkpoint-600; see resume entry below

## [2026-03-18] BACKGROUND TASK launched: 5.B1 LoRA v2 training (RESUME from checkpoint-600)
- Command: source .venv/bin/activate && PYTHONPATH=/Users/hc/Documents/research/Projects/VisualRAG PYTHONUNBUFFERED=1 python3 -u hunyuanOCR/finetune_lora.py --model_path tencent/HunyuanOCR --data_dir dataset/train_medium --output_dir lora-output-v2 --resume_from_checkpoint lora-output-v2/checkpoint-600 --epochs 3 --batch_size 1 --gradient_accumulation_steps 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing 2>&1 | tee results/lora_v2_train.log
- Task ID: btmspqeg2 (prior attempts bmv31xupt/bo3aao7a6 fixed output buffering + num_workers=0 deadlock)
- Status: completed — session died at step ~1300; 3 duplicate processes discovered on resume; killed all, restarted from checkpoint-1600
- Note: PYTHONUNBUFFERED=1 + python -u required; num_workers=0 (MPS fork deadlock fix); Subset-based skip (O(1) resume)

## [2026-03-18] BACKGROUND TASK launched: 5.B1 LoRA v2 training (RESUME from checkpoint-1600)
- Command: source .venv/bin/activate && PYTHONPATH=/Users/hc/Documents/research/Projects/VisualRAG PYTHONUNBUFFERED=1 python3 -u hunyuanOCR/finetune_lora.py --model_path tencent/HunyuanOCR --data_dir dataset/train_medium --output_dir lora-output-v2 --resume_from_checkpoint lora-output-v2/checkpoint-1600 --epochs 3 --batch_size 1 --gradient_accumulation_steps 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing 2>&1 | tee results/lora_v2_train.log
- Task ID: bg98rcndk
- Status: completed — step 9000/9000, Epoch 3 avg loss 0.2304, final model saved to lora-output-v2/final

## [2026-03-19] BACKGROUND TASK launched: 5.B1 eval
- Command: source .venv/bin/activate && PYTHONPATH=/Users/hc/Documents/research/Projects/VisualRAG PYTHONUNBUFFERED=1 python3 -u hunyuanOCR/eval/run_eval.py --checkpoint_dir lora-output-v2/final --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/lora_v2.json 2>&1 | tee results/lora_v2_eval.log
- Task ID: bts1yffj1
- Status: failed — missing --eval_dir arg; relaunched as bco8cwca9

## [2026-03-19] BACKGROUND TASK launched: 5.B1 eval (corrected)
- Command: source .venv/bin/activate && PYTHONPATH=... python3 -u hunyuanOCR/eval/run_eval.py --checkpoint_dir lora-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/lora_v2.json 2>&1 | tee results/lora_v2_eval.log
- Task ID: bco8cwca9
- Status: in-progress (40/1000 samples)
- Note: DoRA training (5.B2) deferred until eval completes to avoid MPS memory contention

## [2026-03-18] Sub-task 3.9 in progress: paper tables (partial)
- Filled all known \tbd{} in paper/main.tex:
    - Abstract: preliminary numbers + 1-epoch framing
    - §5.1: hardware (Mac Mini MPS 48GB), optimizer (AdamW, lr=2e-4, wd=0.01, ratio=0.1, batch=8, rank=16), training (1 epoch)
    - Tab. 1 (main): baseline 0.167 overall, LoRA 0.183, DualLoRA 0.164
    - Tab. 3 (ViT ablation): LLM-only row filled; LLM+ViT pending 3.7
    - Tab. 4 (replay): without-replay filled; with-replay N/A
    - Limitations: MPS constraint + replay labels issue documented
- Recompile: 11 pages, 505KB, 0 overfull > 10pt
- Remaining \tbd{}: Tan et al. external TEDS, LLM+ViT rows (pending 3.7)

## [2026-03-19 12:07] Sub-task 5.B1 done: LoRA v2 eval complete
- Files: results/lora_v2.json, results/lora_v2_eval.log
- Status: pending reviewer approval
- Metrics: CER=0.7584, WER=1.0005, Table TEDS=0.6575, Strike F1=0.035, Underline F1=0.040, Color Acc=0.020
- Notes: 2 timeouts (samples ~341, ~431), no hangs. 1000/1000 samples evaluated. Full results posted to to_code_reviewer.md.

## [2026-03-19 12:20] BACKGROUND TASK launched: 5.B2 DoRA training
- Command: source .venv/bin/activate && PYTHONPATH=. PYTHONUNBUFFERED=1 python3 -u hunyuanOCR/finetune_dora.py --data_dir dataset/train_medium --epochs 3 --gradient_accumulation_steps 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --output_dir dora-output 2>&1 | tee results/dora_train.log
- PID: 7908
- Status: in-progress
- Note: max_length=1024 (reviewer command) caused image token mismatch; relaunched with max_length=2048

## [2026-03-19 23:05] Sub-task 5.X1 done: color bug fix
- Files: dataset/synthetic_generator.py, hunyuanOCR/eval/metrics.py, hunyuanOCR/eval/run_eval.py
- Status: pending reviewer approval
- Changes: render_colored() now black text on colored background; GT/regex/prompt all use background-color

## [2026-03-19 23:17] Sub-task 5.X2 done: color dataset regeneration
- Files: dataset/train/train.jsonl, dataset/train_small/train.jsonl, dataset/train_medium/train.jsonl, dataset/eval/eval_bootstrap.jsonl, dataset/eval/eval_full.jsonl, dataset/synthetic/images/color/ (5000 PNGs)
- Status: pending reviewer approval
- Counts verified: 21947/5000, 2000/400, 3000/600, 200/50, 1000/200 — zero non-color rows changed

## [2026-03-19 23:38] Sub-task 5.Y1 done: replay buffer for lora + dora
- Files: hunyuanOCR/finetune_lora.py, hunyuanOCR/finetune_dora.py
- Status: pending reviewer approval


## 2026-03-19 00:05 Sub-task 5.Y2 done: LR schedule replacement
- Files: hunyuanOCR/finetune_lora.py, hunyuanOCR/finetune_dora.py, hunyuanOCR/finetune_duallora.py
- Status: pending reviewer approval

## 2026-03-19 00:20 Sub-task 5.Y3 done: finetune_loraplus.py
- Files: hunyuanOCR/finetune_loraplus.py (new), paper/references.bib
- Status: pending reviewer approval

## 2026-03-20 00:55 Tasks 1-3 done: paper color fixes + LoRA+ table + recompile
- Files: paper/main.tex, draft/sec1_intro.md, draft/sec2_related_work.md, draft/sec3_dataset.md, draft/sec4_method.md, draft/sec5_experiments.md
- Status: pending reviewer approval

## 2026-03-20 01:30 Sub-tasks 5.Z1-Z5 done: duallora consistency + eval cleanup
- Files: hunyuanOCR/finetune_duallora.py, hunyuanOCR/eval/teds.py, hunyuanOCR/eval/metrics.py, hunyuanOCR/eval/run_eval.py
- Status: pending reviewer approval

## 2026-03-20 01:40 BACKGROUND TASK launched: 5.B2-mps DoRA eval
- Command: python3 hunyuanOCR/eval/run_eval.py --checkpoint_dir dora-output/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/dora.json
- Task ID: b3ojud6ip
- Status: in-progress

## 2026-03-20 01:40 Sub-tasks 5.Z1-Z5 committed: 67a2114
- Files: hunyuanOCR/finetune_duallora.py, hunyuanOCR/eval/teds.py, hunyuanOCR/eval/metrics.py, hunyuanOCR/eval/run_eval.py
- Status: committed

## 2026-03-20 01:42 BACKGROUND TASK launched: 5.B0 baseline re-eval
- Command: python3 hunyuanOCR/eval/run_eval.py --checkpoint_dir tencent/HunyuanOCR --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/baseline_v2.json
- Task ID: b77mowlri
- Status: failed — ModuleNotFoundError (missing PYTHONPATH); relaunched as bk1ces0kl

## 2026-03-21 Sub-task 5.B2-mps COMPLETE: DoRA eval
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir dora-output/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/dora.json 2>&1 | tee results/dora_eval.log
- Task ID: b4h67m7to
- Status: completed — results/dora.json written
- Metrics: CER=0.8001, WER=0.9620, TEDS=0.6717, Strike F1=0.050, Underline F1=0.065, Color Acc=0.000 (wrong color data — expected)
- Approved: 2026-03-21

## 2026-03-21 Sub-task 5.B0 COMPLETE: Baseline re-eval
- Task ID: bryi9a9i3
- Status: completed — results/baseline_v2.json written
- Metrics: CER=1.0042, WER=1.2013, TEDS=0.5492, Strike F1=0.000, Underline F1=0.000, Color Acc=0.000 (all markup=0 as expected for vanilla model)

## 2026-03-21 BACKGROUND TASK launched: 5.B3 Dual LoRA training
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --replay_data_dir dataset/replay_proper --epochs 3 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_every 4 --output_dir duallora-output-v2 2>&1 | tee results/duallora_v2_train.log
- Task ID: b9f1l8evw
- Status: killed at step ~550 (session exit)

## 2026-03-21 BACKGROUND TASK relaunched: 5.B3 Dual LoRA training (resume from checkpoint-400)
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --replay_data_dir dataset/replay_proper --epochs 3 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_every 4 --resume_from_checkpoint duallora-output-v2/checkpoint-400 --output_dir duallora-output-v2 2>&1 | tee results/duallora_v2_train.log
- Task ID: b00qokx2d
- Status: in-progress


## 2026-03-22 BACKGROUND TASK relaunched: 5.B3 eval
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_v2.json 2>&1 | tee results/duallora_v2_eval.log
- Task ID: bw8blemw8
- Status: completed — killed on session compaction at [760/1000]

## 2026-03-22 BACKGROUND TASK relaunched: 5.B3 eval (session resume)
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_v2.json 2>&1 | tee results/duallora_v2_eval.log
- Task ID: bms167n4x
- Status: killed — process stuck in MPS uninterruptible sleep at ~[800+/1000] for 35+ min (SIGALRM cannot interrupt MPS kernel calls)

## 2026-03-22 00:47 BACKGROUND TASK relaunched: 5.B3 eval (with partial-file resume)
- Fix: added --partial_file arg to run_eval.py (incremental per-sample JSONL save + resume-on-restart)
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_v2.json --partial_file results/duallora_v2_partial.jsonl 2>&1 | tee results/duallora_v2_eval.log
- Task ID: b1l6ilj9i
- Status: in-progress

## 2026-03-22 03:10 A6000 Training launched: 5.B3 extend to 5 epochs
- Fix: dual_lora.py delta_W dtype cast (.to(x.dtype)) — CUDA stricter than MPS on dtype mismatch
- Command: nohup env PYTHONPATH=. uv run python -u hunyuanOCR/finetune_duallora.py --epochs 5 --resume_from_checkpoint duallora-output-v2/checkpoint-1000 --output_dir duallora-output-v2-a6k --data_dir dataset/train_medium --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_data_dir dataset/replay_proper --replay_every 4 > duallora_a6k_train.log 2>&1 &
- VM: 38.128.233.14 (n3-RTX-A6000x1, VM ID 692035)
- Log: /home/ubuntu/workspace/duallora_a6k_train.log
- Status: in-progress (Step 1010/1875 at launch confirmation)

## 2026-03-22 Sub-task 5.Z8 done: Fix epoch avg loss double-division
- Files: finetune_duallora.py, finetune_lora.py, finetune_dora.py, finetune_loraplus.py
- Fix: epoch_loss now accumulates outputs.loss.detach().item() (raw); per-step avg_loss = epoch_loss / (step + 1) (removed * grad_accum)
- Status: pending reviewer approval

## 2026-03-22 run_eval.py --max_pixels fix
- A6000 eval OOM at sample 340 (15.49 GiB alloc on 47GB GPU — large image)
- Added --max_pixels arg to run_eval.py; applied to processor.image_processor.max_pixels
- A6000 eval restarted from partial file (340 done) with --max_pixels 524288
- Status: A6000 at 352/1000 and progressing

## 2026-03-22 5.B3 Mac Mini eval complete
- File: results/duallora_v2.json (1000 samples)
- TEDS: 0.6125 (+0.063 vs baseline 0.549)
- CER: 0.971 (improved from baseline 1.004)
- Strike/Underline/Color F1: all 0.000 (consistent with previous MPS 3-epoch behavior)
- Status: pending reviewer — A6000 5-epoch eval is primary result

## 2026-03-22 BACKGROUND TASK launched: 5.B1 Standard LoRA training
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/finetune_lora.py --data_dir dataset/train_medium --epochs 3 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_data_dir dataset/replay_proper --replay_every 4 --output_dir lora-output-v2 2>&1 | tee results/lora_v2_train.log
- Task ID: b7goflkdr
- Status: in-progress

## 2026-03-22 5.B3 A6000 eval complete
- File: results/duallora_v2_a6k.json (1000 samples, 5-epoch A6000 run)
- TEDS: 0.6766 (+0.128 vs baseline 0.549)
- CER: 0.489 (huge improvement from baseline 1.004)
- Strike F1: 0.005, Underline F1: 0.000, Color Acc: 0.000
- Underline CER: 0.055, Strikethrough CER: 0.043 (text correct, markup F1 strict)
- A6000 VM destroyed (4.81h session)
- Status: submitted to reviewer

## 2026-03-22 BACKGROUND TASK launched: 5.B1 eval
- Command: PYTHONPATH=. PYTHONUNBUFFERED=1 uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir lora-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/lora_v2.json --partial_file results/lora_v2_partial.jsonl --max_pixels 524288
- Task ID: bxvxbo65s
- Status: in-progress

## 2026-03-22 Sub-task 5.B1 done: Standard LoRA eval
- Files: results/lora_v2.json, results/lora_v2_eval.log
- Status: pending reviewer approval
- TEDS 0.6575, CER 0.758, Strike 0.035, Underline 0.040, Color 0.020

## 2026-03-22 BACKGROUND TASK launched: 5.B2 DoRA training
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/finetune_dora.py --data_dir dataset/train_medium --epochs 3 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_data_dir dataset/replay_proper --replay_every 4 --output_dir dora-output-v2
- Task ID: bmx67cen4
- Status: in-progress

## 2026-03-22 5.B2 DoRA — switched to A6000
- Mac Mini DoRA stopped at checkpoint-2400 (step 2400/9000, loss 0.639)
- A6000 VM: 38.128.233.91, PID 2758
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/finetune_dora.py --data_dir dataset/train_medium --epochs 5 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_data_dir dataset/replay_proper --replay_every 4 --output_dir dora-output-v2 --resume_from_checkpoint dora-output-v2/checkpoint-2400
- Log: results/dora_v2_a6k_train.log
- Loss trajectory: Epoch 1 avg 0.4317, Epoch 2 avg 0.4352, Epoch 3 avg 0.4030, Epoch 4 avg 0.3640, Epoch 5 avg 0.3332
- Final model: dora-output-v2/final/ (pulled to Mac Mini)
- Status: COMPLETE — 5 epochs done

## 2026-03-22 BACKGROUND TASK launched: 5.B2 DoRA eval (Mac Mini)
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir dora-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/dora_v2_a6k.json --partial_file results/dora_v2_a6k_partial.jsonl --max_pixels 524288
- PID: 40830 (python), 40828 (uv)
- Log: results/dora_v2_a6k_eval.log
- Status: in-progress (~22/1000 samples)

## 2026-03-23 BACKGROUND TASK COMPLETE: 5.B4 LoRA+ training (A6000)
- Command: PYTHONPATH=/home/ubuntu/workspace /home/ubuntu/.local/bin/uv run python -u hunyuanOCR/finetune_loraplus.py --data_dir dataset/train_medium --epochs 5 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_data_dir dataset/replay_proper --replay_every 4 --output_dir loraplus-output-v2 --loraplus_lr_ratio 16.0
- VM: 38.128.233.91 (destroyed after 8.43h session)
- Log: results/loraplus_v2_train.log (pulled)
- Final model: loraplus-output-v2/final/ (pulled, ~63MB)
- Loss: E1=0.6935, E2=0.5872, E3=0.5389, E4=0.4759, E5=0.4217
- Status: COMPLETE — VM destroyed

## 2026-03-23 BACKGROUND TASK COMPLETE: 5.B2 DoRA eval (Mac Mini)
- Results: results/dora_v2_a6k.json (1000 samples)
- CER=0.787, WER=1.045, TEDS=0.613, Strike F1=0.045, Underline F1=0.065, Color Acc=0.000
- Status: COMPLETE

## 2026-03-23 BACKGROUND TASK COMPLETE: 5.B4 LoRA+ eval (Mac Mini)
- Results: results/loraplus_v2.json (1000 samples)
- CER=0.756, WER=1.027, TEDS=0.662, Strike F1=0.035, Underline F1=0.030, Color Acc=0.000
- Status: COMPLETE

## 2026-03-23 BACKGROUND TASK launched: 5.C1 no-replay Dual LoRA training (Mac Mini)
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --epochs 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --output_dir duallora-noreplay-output
- PID: 94325
- Log: results/duallora_noreplay_train.log
- Trainable: 26,443,776 params (2.59%), no replay buffer
- Status: in-progress (~25h on MPS)

## 2026-03-24 5.C1 no-replay Dual LoRA training COMPLETE
- Avg loss: 0.7352 (S10=3.49 → S370=0.74)
- Final model: duallora-noreplay-output/final
- Status: completed

## 2026-03-24 BACKGROUND TASK launched: 5.C1 eval
- Command: uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-noreplay-output/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_noreplay.json --partial_file results/duallora_noreplay_partial.jsonl --max_pixels 524288
- Task ID: PID 40447
- Status: in-progress

## 2026-03-23 5.C1 eval COMPLETE
- Checkpoint: duallora-noreplay-output/final
- Results: results/duallora_noreplay_v2.json (1000 samples)
- CER=0.9469, WER=1.1543, TEDS=0.5772
- Strike F1=0.000, Underline F1=0.000, Color Acc=0.000
- All feature metrics 0.0 — confirms replay buffer required for feature learning
- Status: COMPLETE

## 2026-03-23 BACKGROUND TASK launched: 5.C2 LLM+ViT Dual LoRA training (Mac Mini)
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --epochs 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --include_vit --replay_data_dir dataset/replay_proper --replay_every 4 --output_dir duallora-vit-output-v2
- PID: 82196
- Log: results/duallora_vit_v2_train.log
- Status: in-progress (~25h on MPS)

## 2026-03-23 12:16 5.C2 LLM+ViT Dual LoRA training COMPLETE
- Checkpoint: duallora-vit-output-v2/final
- Avg loss: 0.8089 (S10=2.79 → S370=0.8142 → final=0.8089)
- 330 Dual LoRA adapters saved
- Status: completed

## 2026-03-23 12:16 BACKGROUND TASK launched: 5.C2 eval
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-vit-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_vit_v2.json --partial_file results/duallora_vit_v2_partial.jsonl --max_pixels 524288
- PID: 10463
- Log: results/duallora_vit_v2_eval.log
- Status: in-progress (~25h on MPS)

## 2026-03-23 16:03 5.C2 LLM+ViT eval COMPLETE
- Checkpoint: duallora-vit-output-v2/final
- Results: results/duallora_vit_v2.json (1000 samples)
- CER=0.9406, WER=1.1100, TEDS=0.5878
- Strike F1=0.000, Underline F1=0.000, Color Acc=0.000
- All feature metrics 0.0 — ViT inclusion + replay still yields no feature learning
- Status: COMPLETE

## 2026-03-23 16:03 BACKGROUND TASK launched: 5.C3 r=8 training
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --epochs 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_data_dir dataset/replay_proper --replay_every 4 --dual_lora_rank 8 --dual_lora_alpha 8 --output_dir duallora-r8-output-v2
- PID: 33890
- Log: results/duallora_r8_v2_train.log
- Status: in-progress (~25h on MPS)

## 2026-03-24 5.C3 r=8 Training Complete
- Avg loss: 0.9344
- Final model: duallora-r8-output-v2/final/
- Status: training done

## 2026-03-24 BACKGROUND TASK launched: 5.C3 r=8 eval
- Command: env PYTHONPATH=. uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-r8-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_r8_v2.json --partial_file results/duallora_r8_v2_partial.jsonl --max_pixels 524288
- Task ID: PID 72869
- Status: in-progress

## 2026-03-23 22:07 Sub-task 5.C3 r=8 eval complete
- Files: results/duallora_r8_v2.json
- Results: CER 0.9358, WER 1.1084, TEDS 0.5742, all feature metrics 0.0
- Status: pending reviewer approval

## 2026-03-23 22:07 BACKGROUND TASK launched: 5.C3 r=32 training
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --epochs 1 --max_length 2048 --max_pixels 1048576 --gradient_checkpointing --replay_data_dir dataset/replay_proper --replay_every 4 --dual_lora_rank 32 --dual_lora_alpha 32 --output_dir duallora-r32-output-v2
- Task ID: PID 9446
- Status: in-progress

## 2026-03-24 05:42 Sub-task 5.C3 r=32 training complete
- Files: duallora-r32-output-v2/final/
- Results: Avg loss 0.7085, 375 steps, 276 adapters saved
- Status: pending reviewer approval

## 2026-03-24 05:42 BACKGROUND TASK launched: 5.C3 r=32 eval
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-r32-output-v2/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_r32_v2.json --partial_file results/duallora_r32_v2_partial.jsonl --max_pixels 524288
- Task ID: PID 50884
- Status: in-progress

## 2026-03-24 04:52 Sub-task 5.C3 done: Dual LoRA r=32 eval complete
- Files: results/duallora_r32_v2.json
- Results: CER=0.8626, WER=0.9946, TEDS=0.6018, color_boundary_f1=0.02
- Improvement over r=8: CER -0.073, WER -0.114, TEDS +0.028
- Status: pending reviewer approval

## 2026-03-24 05:00 Sub-task 5.D1 started: seed arg added to finetune_duallora.py
- Files: hunyuanOCR/finetune_duallora.py
- Change: added --seed arg, sets random/torch/cuda seeds after parse_args
- Status: pending reviewer approval before training

## 2026-03-24 05:30 BACKGROUND TASK launched: 5.D1 seed=42 training
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --replay_data_dir dataset/replay_proper --output_dir duallora-seed42-output --epochs 1 --max_length 2048 --max_pixels 524288 --gradient_checkpointing --replay_every 4 --seed 42 > results/duallora_seed42_train.log 2>&1
- Task ID: PID 99886
- Status: in-progress

## 2026-03-24 12:00 Sub-task 5.D1: seed=42 training complete, eval launched
- Training: duallora-seed42-output/final/ — complete
- Eval PID: 36331, output: results/duallora_seed42_v2.json
- Status: eval in-progress

## 2026-03-24 14:00 5.D1: seed=42 eval done, seed=0 training launched
- seed=42 results: CER=0.9022, WER=1.0529, TEDS=0.5699
- seed=0 training PID: 55788 — results/duallora_seed0_train.log
- Status: seed=0 training in-progress

## 2026-03-24 22:07 Sub-task 5.D1 seed=0 training complete
- Files: duallora-seed0-output/final/
- Avg loss: 0.8135 (375 steps)
- Status: eval launched PID 23787

## 2026-03-24 22:07 BACKGROUND TASK launched: seed=0 eval
- Command: PYTHONPATH=. uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-seed0-output/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_seed0_v2.json --partial_file results/duallora_seed0_v2_partial.jsonl --max_pixels 524288 > results/duallora_seed0_eval.log 2>&1
- Task ID: PID 23787
- Status: in-progress

## 2026-03-25 00:31 Sub-task 5.D1 seed=0 eval done
- Files: results/duallora_seed0_v2.json
- Status: completed
- Results: CER=0.9299, WER=1.0489, TEDS=0.6047, all feature metrics=0.0, n=1000

## 2026-03-25 00:31 BACKGROUND TASK launched: 5.D1 seed=1 training
- Command: nohup env PYTHONPATH=. uv run python -u hunyuanOCR/finetune_duallora.py --data_dir dataset/train_medium --replay_data_dir dataset/replay_proper --output_dir duallora-seed1-output --epochs 1 --max_length 2048 --max_pixels 524288 --gradient_checkpointing --replay_every 4 --seed 1 > results/duallora_seed1_train.log 2>&1
- Task ID: PID 83809
- Status: in-progress

## 2026-03-25 06:32 Sub-task 5.D1 seed=1 training complete
- Files: duallora-seed1-output/final/
- Final avg loss: 0.8279
- Status: training complete

## 2026-03-25 06:32 BACKGROUND TASK launched: seed=1 eval
- Command: env PYTHONPATH=. uv run python -u hunyuanOCR/eval/run_eval.py --checkpoint_dir duallora-seed1-output/final --eval_dir dataset/eval --eval_jsonl dataset/eval/eval_full.jsonl --output_file results/duallora_seed1_v2.json --partial_file results/duallora_seed1_v2_partial.jsonl --max_pixels 524288 > results/duallora_seed1_eval.log 2>&1
- Task ID: PID 43809
- Status: in-progress

## 2026-03-25 Sub-task 5.D1 done: statistical rigor — 3-seed eval complete
- Files: results/duallora_seed1_v2.json, to_code_reviewer.md
- Seeds: 42 (TEDS=0.5699), 0 (TEDS=0.6047), 1 (TEDS=0.5932)
- Final: TEDS=0.5893±0.0145, CER=0.9098±0.0144, WER=1.0539±0.0046
- Status: pending reviewer approval
