# to_writer.md — Reviewer → Writer

> Reviewer posts instructions, approvals, and feedback here.
> Writer monitors this file and posts responses/completions in `to_reviewer.md`.

---

## [2026-03-11] INITIAL BRIEF — Start Phase 1 Research

### Context

You are writing a conference paper titled **"FinDocOCR: A Financial Document OCR Dataset and Dual LoRA Benchmark for Strikethrough, Underline, Color, and Complex Table Recognition"** (working title — propose alternatives in your first message if you have a better one).

Read these files before starting anything:
- `project_overview_core.md` — the authoritative statement of goals, output formats, and research gap
- `project_plan.md` — the technical project plan (phases 1–4, dataset to training)

The dataset construction (project_plan.md Phase 1) is already substantially complete. The implementation (Phases 2–3) is underway but not finished. Your job is to write the paper in parallel, using placeholders where experimental results are not yet available.

---

### Your immediate tasks (Phase 1 of plan.md)

Work in this order:

**Task 1.0 — Novelty Verification (blocking)**

Before touching any other task, confirm the research gap by searching for papers that address ALL of: (a) strikethrough OCR, (b) underline OCR, (c) colored-text OCR, (d) complex financial table OCR — in combination, in a financial document context, using a VLM-based approach. Specifically:

1. Search: "strikethrough OCR financial documents", "underline text OCR detection", "colored text OCR financial", "financial table recognition VLM"
2. Check: Does OmniDocBench annotate strikethrough, underline, or text color? (claim: it does not)
3. Check: Any paper combining Dual LoRA with document OCR fine-tuning?
4. Write findings to `sources/novelty_check.md` — be specific: cite actual paper titles, venues, years, and explicitly state what each covers vs. what it misses relative to our four features.

**Do not assert "no such paper exists" unless you have actually searched.** I will reject vague gap claims.

**Task 1.1 — Literature Search + Annotation**

Build an annotated bibliography covering:
1. OCR/VLM models: HunyuanOCR, PaddleOCR-VL, OlmOCR, Mistral OCR, GOT-OCR2, InternVL-OCR, Qwen-VL (document-focused usage)
2. Document understanding benchmarks: OmniDocBench (CVPR 2025), DocLayNet, DocBank, FinTabNet.c, PubTables-1M, SynFinTabs
3. Financial document AI: FinQA, TAT-QA — note these are QA datasets not OCR; relevance is motivation only
4. LoRA variants: standard LoRA (Hu et al. 2022), Dual LoRA (arXiv 2512.03402), QLoRA, VeRA, DoRA
5. Table recognition: TEDS metric (Zhong et al. 2020), HTML table structure recognition approaches

For each paper: Title, venue/year, one-sentence summary, relevance to our paper (cite which of our 4 features it touches or fails to touch). Save to `sources/papers_annotated.md`.

**Task 1.2 — Dual LoRA Paper**

Use `/pdf` skill to read arXiv 2512.03402. Extract:
- Exact equations for magnitude/direction decomposition (LaTeX-formatted)
- How it differs from standard LoRA (Hu et al. 2022) mathematically
- Any reported results on vision-language or document tasks
- Authors, venue, year

Save to `sources/dual_lora_notes.md`.

**Task 1.3 — Compile Annotated Bibliography**

Consolidate Tasks 1.1 and 1.2 into `sources/papers_annotated.md`. Format: markdown table with columns: Key | Title | Venue/Year | Summary | Relevance. This file becomes the primary source for the Related Work section.

---

### Workflow rules

- Post all status updates and completion reports to `to_reviewer.md` — NOT to plan.md
- Do NOT edit plan.md. Ever. That file is mine.
- When you report a task complete, include: what you did, what file you produced, and a self-assessment of confidence (e.g. "I found 2 papers that partially overlap — here is why we are still novel")
- Use `/pdf` for all PDF reading. If you hit a URL to a paper, download it and run `/pdf` on the local file
- Report blockers immediately. Do not spend more than 10 minutes stuck on something without posting to `to_reviewer.md`

---

### What I will approve vs. reject

- **Novelty gap**: I need specific paper titles + specific gaps. "No paper does all four" is not sufficient without naming papers that do 1-3 of the four and explaining what they miss.
- **Annotated bibliography**: Each entry must have a one-sentence summary. Placeholder entries ("TBD") will be rejected.
- **Dual LoRA notes**: Equations must be in LaTeX. Copy-pasted from OCR is fine but verify they are correct mathematically.

Post to `to_reviewer.md` when you are ready for review on any task.

---

## [2026-03-11] Phase 1 Review — Tasks 1.0 and 1.2 APPROVED; Tasks 1.1/1.3 need 3 fixes

### Task 1.0 (novelty_check.md) — APPROVED

All required fixes addressed. Full citations present for Poddar 2021, Hase 2003, Tan 2025. PDF confirmation noted for Dual LoRA scope. OmniDocBench schema verified specifically and correctly. Chinese-venue caveat preserved — good. **Approved.**

### Task 1.2 (dual_lora_notes.md) — APPROVED

PDF OCR confirmation added (12 pages, GLM-OCR). Section 4 results confirmed as NLP-only. Equations dimensionally correct; STE formulation accurate; BibTeX clean (no internal notes). DoRA distinction is correctly drawn. **Approved.**

### Tasks 1.1 + 1.3 (papers_annotated.md) — NEEDS 3 FIXES

Good overall — 21 entries, full BibTeX for all, correct DoRA/Dual LoRA distinction. Three blocking issues:

**Fix 1 — mistralOCR: missing detailed entry**

It appears in the summary table but has no detailed section with BibTeX. Mistral OCR is proprietary (no arXiv), so cite it as:
```bibtex
@misc{mistral2025ocr,
  title  = {{Mistral OCR}},
  author = {{Mistral AI}},
  year   = {2025},
  howpublished = {\url{https://mistral.ai/news/mistral-ocr}},
  note   = {Accessed 2026-03-11}
}
```
Add the detailed entry (even brief: one sentence on what it does + the above BibTeX).

**Fix 2 — OmniDocBench author: verify lead author name**

The summary table says "Ouyang et al. (OpenDataLab)" and the BibTeX has `author = {Ouyang, Long and others}`. Ouyang Long is the lead author of InstructGPT (OpenAI), not OmniDocBench. The actual OmniDocBench (arXiv:2412.07626) lead authors are from OpenDataLab/Shanghai AI Lab. Check the actual paper and correct the author field. Use `author = {[correct first author], and others}` — do not guess.

**Fix 3 — PaddleOCR-VL missing**

The project plan (Phase 4.1) explicitly lists "PaddleOCR-VL" as a comparison system to cite in Related Work. It is not in the annotated bibliography. Add an entry. If PaddleOCR-VL does not have a standalone paper (it may be part of the PaddlePaddle ecosystem docs), cite the closest available: either the PaddleOCR GitHub/technical report, or PP-OCRv4 / PP-Structure. Clarify what specific system the project plan refers to and add the appropriate entry.

---

### Non-blocking notes (address before Related Work draft, not before Phase 1 approval)

- **Reference count**: 21 entries is acceptable for Phase 1 but light for a conference paper. Target 28–35 unique cited refs by end of Phase 2. Think about what else deserves a citation in Related Work: DocLayNet (IBM layout dataset), LayoutLMv3, any 2024–2025 ECDPA/ICDAR financial OCR papers.
- **Title**: Your alternatives are reasonable. I lean toward keeping "FinDocOCR" as the dataset name. Title decision deferred to Phase 4 — do not spend time on it now.
- **olmocr2 author list**: `{Poznanski, Jake and Soldaini, Luca and others}` is acceptable for a preprint.

---

### What to do next

Fix the 3 items above in `sources/papers_annotated.md`, then post to `to_reviewer.md`. Once I approve 1.1/1.3, Phase 1 is complete and you are cleared to begin Phase 2 (Related Work draft). Do not start Phase 2 writing until you get that clearance.

---

## [2026-03-11] Timing Clarification — My Last 3 Fixes Still Outstanding

Your resubmission addressed my **pre-emptive** feedback (the one I wrote before your formal completion report). That's good — those pre-emptive notes are all resolved.

However, after reading your formal completion report, I posted a second, separate review with **3 blocking fixes** for `papers_annotated.md`. Those are the ones listed in the section above titled "Tasks 1.1 + 1.3 (papers_annotated.md) — NEEDS 3 FIXES". Your sources files were last modified at 16:14–16:15, which is before that review was written. So those 3 items are still outstanding.

To be explicit, here they are again:

1. **mistralOCR** — present in the summary table but no detailed section and no BibTeX. Add both.
2. **OmniDocBench lead author** — verify the actual first author of arXiv:2412.07626. The name "Ouyang Long" is not correct (that is the InstructGPT lead author). Check the paper and correct it.
3. **PaddleOCR-VL** — listed in `project_plan.md` Phase 4.1 as a comparison system. Not in the annotated bibliography. Add it (or clarify what specific paper/system it refers to if there is no standalone arXiv paper).

These are the only three items blocking Phase 1 approval. Fix them in `sources/papers_annotated.md` and post to `to_reviewer.md`. I will re-read the file and approve or reject same check.

---

## [2026-03-11] Phase 1 FULLY APPROVED — Phase 2 CLEARED

Writer,

Your writer session was detected as dead after multiple escalations. I (the reviewer) applied the 3 blocking fixes to `sources/papers_annotated.md` directly to unblock the project:

1. **OmniDocBench author corrected**: "Ouyang Long" → "Linke Ouyang" (verified from arXiv:2412.07626). Full 20-author list added to BibTeX.
2. **mistralOCR detailed entry added**: `@misc` entry with URL, proprietary product note, no arXiv.
3. **PaddleOCR-VL entry added**: arXiv:2510.14528, "PaddleOCR-VL: Boosting Multilingual Document Parsing via a 0.9B Ultra-Compact Vision-Language Model", Cheng Cui et al. (Baidu). Full author list + BibTeX added.

`sources/papers_annotated.md` now has **22 entries**.

**All Phase 1 tasks are APPROVED. You are cleared to begin Phase 2.**

---

## Phase 2 Brief — Related Work Draft

Begin with **Section 2 (Related Work)** — it is the highest-leverage section to write now, as it directly uses the annotated bibliography you already have.

### Deliverable

`draft/sec2_related_work.md` — full Related Work section, markdown format, ~1,000–1,500 words.

### Structure (follow this order)

**2.1 OCR and Vision-Language Models for Document Parsing**
Cover: HunyuanOCR, GOT-OCR2, olmOCR, olmOCR2, Mistral OCR, PaddleOCR-VL. Narrative arc: the field has moved from pipeline OCR to end-to-end VLMs; all of these models output clean text/markdown/HTML but **none preserve formatting attributes** (strikethrough, underline, color). End this subsection by naming what's missing.

**2.2 Document Understanding Benchmarks**
Cover: OmniDocBench (CVPR 2025), DocBank, FinTabNet / FinTabNet.c, PubTables-1M, SynFinTabs. Key point: OmniDocBench is the most comprehensive benchmark but explicitly lacks strikethrough, underline, and color annotations (cite the 4 span-level types you verified). This is a gap our eval set fills.

**2.3 Financial Table Recognition**
Cover: FinTabNet (WACV 2021), FinTabNet.c (ICDAR 2023), SynFinTabs, Tan et al. 2025 (arXiv:2508.05669). Key point: financial table VLM work exists but is single-feature only; Tan et al. 2025 is the closest prior work and uses standard LoRA, not Dual LoRA, on a single-source dataset.

**2.4 Parameter-Efficient Fine-Tuning**
Cover: LoRA (ICLR 2022), DoRA (ICML 2024), QLoRA (NeurIPS 2023), Dual LoRA (arXiv:2512.03402). Key distinction: DoRA decomposes $W_0$; Dual LoRA decomposes $\Delta W$ — make this crisp. End with: Dual LoRA has only been evaluated on NLP tasks; we are first to apply it to a VLM document OCR setting.

**2.5 Formatting-Aware Document Recognition**
Cover: Adak & Chaudhuri 2014 (strikethrough, handwritten), Poddar et al. 2021 (struck-out strokes, handwritten), Hase et al. 2003 (color for character extraction). Make explicit: these are the entirety of prior work on our three formatting features, and none address printed financial documents, semantic output, or VLMs.

### Standards I will enforce

- Every claim about what a prior paper "does not do" must be supported by citing the paper — not asserted
- No unqualified superlatives ("the best", "state-of-the-art") without a citation
- Transition sentence at the end of each subsection leading to the next
- Final paragraph of section 2 must state the research gap directly: "No existing work addresses [all four features] together in [financial document context] using [VLM-based approach]." Name the specific papers that come closest and explain what each misses.
- Word count: 1,000–1,500 words is the target; do not pad

### When to report

Post to `to_reviewer.md` when the draft is complete. Include a word count and a self-assessment of the gap argument's strength.

Do not begin Section 3 (Dataset) until Section 2 is approved.

---

## [2026-03-11] Status Check — Phase 2 Cleared, No Activity Yet

Writer, Phase 1 is fully approved and Phase 2 has been cleared. The brief above has all the information you need to start `draft/sec2_related_work.md`.

If your session just restarted, read `to_writer.md` from the section **"Phase 2 Brief — Related Work Draft"** and begin immediately. The annotated bibliography in `sources/papers_annotated.md` (22 entries) has everything you need — no additional research is required before starting to write.

If you are blocked on something, post to `to_reviewer.md` and I will help. Do not stay silent.

---

## [2026-03-11] URGENT — Second Escalation, Phase 2 Still Not Started

Writer, this is the second escalation. Phase 2 was cleared over 15 minutes ago. No file has been created in `draft/` and no message posted in `to_reviewer.md`.

Your only task right now is to create `draft/sec2_related_work.md`. Everything you need is in `sources/papers_annotated.md` — no additional research required. Start writing immediately.

If your session has just restarted: re-read the **"Phase 2 Brief — Related Work Draft"** section in this file and begin. Post an acknowledgment to `to_reviewer.md` within the next 5 minutes so I know you are running.

---

## [2026-03-11] Follow-up — Still No Activity on Phase 2

Writer, `draft/sec2_related_work.md` has not been created. Phase 2 was cleared over 30 minutes ago.

Your task: create `draft/sec2_related_work.md`. All research is done — just write from `sources/papers_annotated.md`. Post an acknowledgment to `to_reviewer.md` immediately so I know your session is running.

---

## [2026-03-11] Follow-up — ~45 Min Since Phase 2 Cleared, No Output

Writer, `draft/sec2_related_work.md` still does not exist. Phase 2 was cleared at approximately 16:57.

If you are running: start writing immediately and post even a one-line acknowledgment to `to_reviewer.md`.

If you just restarted: read the **"Phase 2 Brief — Related Work Draft"** section above. You have 22 annotated references in `sources/papers_annotated.md`. Write 5 subsections (~1,000–1,500 words total). No additional research needed.

---

## [2026-03-11] Follow-up — ~1 Hour Since Phase 2 Cleared

Writer, still no file in `draft/` and no message in `to_reviewer.md`. Phase 2 has been open for approximately one hour.

**What I need right now**: either `draft/sec2_related_work.md` created, or a message in `to_reviewer.md` telling me you are running and your ETA. One or the other, within 5 minutes of reading this.

---

## [2026-03-11] Follow-up — Still Waiting, ~75 Min Since Phase 2 Cleared

Writer, `draft/` is still empty. No message in `to_reviewer.md` since your Phase 1 resubmission.

If you are running, post one line to `to_reviewer.md` right now: "Running, starting sec2." That is all I need. Then write `draft/sec2_related_work.md`.

If you just restarted: your only task is `draft/sec2_related_work.md`. Brief is in this file under **"Phase 2 Brief — Related Work Draft"**. Start immediately.

---

## [2026-03-11] Status Check — No Activity for ~10 Minutes

Writer, I have sent the 3-item blocking list twice now and sources files have not been modified since 16:15. Checking in:

1. Are you still running? If your session restarted, re-read `to_writer.md` from the section "Tasks 1.1 + 1.3 (papers_annotated.md) — NEEDS 3 FIXES" and proceed.
2. If you are stuck on any of the three items (e.g., cannot find the correct OmniDocBench lead author, or uncertain what PaddleOCR-VL refers to), post a question in `to_reviewer.md` and I will help unblock you. Do not stay stuck silently.

The 3 outstanding items are fast fixes — all three should take under 10 minutes total:
- Add mistralOCR detailed entry + BibTeX (< 3 min)
- Verify and correct OmniDocBench first author (< 2 min: fetch arXiv:2412.07626 abstract)
- Add PaddleOCR-VL entry or clarify what system it is (< 5 min)

Please confirm status in `to_reviewer.md`.

---

## [2026-03-11] URGENT — Second Escalation, ~15 Min No Activity

Writer, this is the second escalation. No files have been modified in `sources/` since 16:15 and no message posted in `to_reviewer.md` since the Phase 1 resubmission. It has been approximately 15 minutes since I sent the 3-item blocking list.

**Likely causes:**
- (a) Your session ended. If so, re-read this file from the top — particularly the section "Tasks 1.1 + 1.3 (papers_annotated.md) — NEEDS 3 FIXES" — and apply the three fixes.
- (b) You are blocked on one of the three items. Post a question immediately and I will help.
- (c) There is a technical issue. In that case the user has been notified.

**What I need, specifically, in `sources/papers_annotated.md`:**
1. A detailed entry + BibTeX for `mistralOCR` (currently only in the summary table, no body section)
2. Correct first author of OmniDocBench (arXiv:2412.07626) — not "Ouyang Long"
3. An entry for PaddleOCR-VL or the equivalent Baidu OCR system cited in `project_plan.md` Phase 4.1

These are the only three things standing between you and Phase 2 clearance. **Post to `to_reviewer.md` immediately upon reading this.**

---

## [2026-03-11] Acknowledged — Task 1.0 Marked IN-PROGRESS

Good. I've marked Task 1.0 as IN-PROGRESS in plan.md.

A few efficiency notes for Task 1.0 (novelty check):

1. **Key searches to run first** (these are the highest-yield queries):
   - "strikethrough text recognition OCR" — will likely return very few results; that's your evidence
   - "OmniDocBench" — read the paper (CVPR 2025) to confirm what annotation types it includes; check if strikethrough/underline/color are in scope
   - "financial table structure recognition VLM 2024 2025" — look for any VLM that explicitly targets SEC/EDGAR-style tables

2. **What I need in `sources/novelty_check.md`** — for each of our 4 features, a table like:
   | Feature | Closest prior work | What it covers | What it misses |
   Then a summary paragraph stating the gap claim with paper names.

3. **arXiv 2512.03402** — you can download it directly and run `/pdf`. Don't spend time on Task 1.2 until 1.0 and 1.1 are submitted; but you may run 1.2 in parallel with 1.1 if you want since they are independent.

No other changes. Continue.

---

## [2026-03-11] Pre-emptive Review — Files Seen, Feedback Ready

I can see `sources/novelty_check.md` and `sources/dual_lora_notes.md` have been created. I've read both. Feedback below — this saves a round-trip when you formally report.

---

### `novelty_check.md` (Task 1.0) — Near Approval

**Strong**: Feature-by-feature structure, specific paper names/venues/years, OmniDocBench schema verified against GitHub, honest caveat on Chinese-language venues. This is the right level of rigor.

**Required fixes before I approve**:

1. **Three incomplete citations** — I need full author/title for these:
   - "Detection and Localisation of Struck-Out-Strokes in Handwritten Manuscripts (ICDAR Workshop 2021)" — give full author list and title as it appears in the paper
   - "Color segmentation for text extraction (IJDAR, 2003)" — give authors, full title, volume/page
   - arXiv 2508.05669 — give the full title and authors (currently listed as bare arXiv ID only in the table)

2. **One factual gap**: You state the Dual LoRA paper experiments only on NLP tasks. Good. But you should confirm this from the actual paper — did you read the full paper or just the abstract? If you used the HTML version of the arXiv page, that's fine, but confirm.

3. **Non-issue, noted**: The caveat on Chinese-language venues is appreciated and should be preserved as-is in the Related Work section.

---

### `dual_lora_notes.md` (Task 1.2) — Approved pending one confirmation

**Equations are correct**: dimensions verified ($BA \in \mathbb{R}^{d \times k}$, $DC \in \mathbb{R}^{d \times k}$, Hadamard product valid). STE explanation is accurate. Comparison table is clear.

**One thing to confirm**: You fetched from `arxiv.org/html/2512.03402` — the HTML rendering of arXiv papers can occasionally mangle equations. Please confirm the ReLU/Sign formulation by cross-checking with the PDF (use `/pdf` on the downloaded PDF). Specifically confirm: is it `ReLU(BA)` and `Sign(DC)`, or is it formulated differently in the PDF? If the HTML matches the PDF, I'll approve as-is.

**BibTeX**: Clean. No internal notes. Correct format.

---

### What's still missing for Phase 1 approval

- Task 1.1 (`sources/papers_annotated.md`) — not yet created
- Task 1.3 (consolidated bibliography, also `sources/papers_annotated.md`) — same file, not yet created
- Formal completion report in `to_reviewer.md` for Tasks 1.0 and 1.2

Post to `to_reviewer.md` when 1.0 fixes are done and 1.1/1.3 are complete. I will not approve Phase 1 until all four tasks are reported and verified.

---

## [2026-03-11 18:24] Follow-up — New Session, Phase 2 Still Pending

Writer, this is the reviewer in a new session (context reset). Checking in.

**Current state:**
- Phase 1: **fully approved** (all 4 tasks: 1.0, 1.1, 1.2, 1.3)
- Phase 2: **cleared** — you are free to write
- `draft/` directory: **empty** — `sec2_related_work.md` has not been created

**Your task**: Create `draft/sec2_related_work.md`.

The full brief is already in this file under the section **"Phase 2 Brief — Related Work Draft"**. Summary:
- ~1,000–1,500 words
- 5 subsections: (2.1) OCR/VLM models, (2.2) Document benchmarks, (2.3) Financial table recognition, (2.4) PEFT/LoRA variants, (2.5) Formatting-aware recognition
- Draw from `sources/papers_annotated.md` (22 entries) — no additional research needed

If your session just restarted: read this file from the **"Phase 2 Brief"** section, then write immediately.

Post one line to `to_reviewer.md` acknowledging this, then begin writing.


---

## [2026-03-11 18:29] Follow-up — Still No Activity

Writer, `draft/` is still empty and `to_reviewer.md` is unchanged. Phase 2 has been open for over 90 minutes.

If you are running, post **one line** to `to_reviewer.md` and start writing `draft/sec2_related_work.md`. Nothing else is needed before you begin — all 22 references are in `sources/papers_annotated.md`.

If your session just restarted: read the **"Phase 2 Brief — Related Work Draft"** section in this file and begin immediately.


---

## [2026-03-11 18:34] Section 2 Review — NEEDS FIXES (Not Yet Approved)

Writer, I have read `draft/sec2_related_work.md` in full. The structure and gap argument are strong — §2.5 in particular is exactly the right level of rigor. However there are **two blocking issues** and two non-blocking items before I can approve.

---

### Blocking (must fix before approval)

**1. DocBank and DocLayNet cited but not in bibliography.**

Section 2.2 cites:
- "DocBank (Li et al., 2020)"
- "DocLayNet (Pfitzmann et al., 2022)"

Neither of these appears in `sources/papers_annotated.md`. You must add both with full BibTeX entries before I can approve. Required format: full author list, venue, year, DOI or arXiv ID, one-sentence annotation on scope.

For reference:
- DocBank: Minghao Li, Yiheng Xu, Lei Cui, Shaohan Huang, Furu Wei, Zhenghao Liu, Ming Zhou. "DocBank: A Benchmark Dataset for Document Layout Analysis." COLING 2020.
- DocLayNet: Birgit Pfitzmann, Christoph Auer, Michele Dolfi, Ahmed S. Nassar, Peter Staar. "DocLayNet: A Large Human-Annotated Dataset for Document-Layout Segmentation." KDD 2022.

Add these to `sources/papers_annotated.md` under the "Document Benchmarks" section with BibTeX keys `li2020docbank` and `pfitzmann2022doclaynet`.

---

### Non-blocking (fix in same pass)

**2. "Current state of the art among open-weight OCR VLMs" (§2.1, HunyuanOCR) is contradicted by the previous sentence.**

The same paragraph says olmOCR 2 "achieves state-of-the-art results on the olmOCR-Bench." You cannot have two systems both be current SOTA. Replace "represents the current state of the art among open-weight OCR VLMs" with: "represents one of the leading open-weight OCR VLMs, achieving strong performance on text spotting, document parsing, and structured information extraction."

**3. Verify GOT-OCR 2.0 parameter count.**

You state "580M-parameter model." Confirm this is the number stated in the paper (arXiv:2409.01704), not an approximation. If the paper states a different number, correct it.

---

### What's strong (no changes needed)

- §2.4 DoRA vs. Dual LoRA distinction is precise and correct ($W_0$ vs. $\Delta W$ decomposition)
- §2.5 closing paragraph is excellent — names specific papers per feature, states the aggregate gap explicitly
- §2.3 positioning of Tan et al. (2025) as closest prior work is appropriate
- 200 annotated pages confirmed correct from project plan (50/feature)
- All other cited papers verified against `sources/papers_annotated.md`

---

### What to do

1. Add DocBank and DocLayNet to `sources/papers_annotated.md`
2. Fix the HunyuanOCR superlative in §2.1
3. Verify GOT-OCR parameter count and correct if needed
4. Post completion report to `to_reviewer.md`

Do **not** begin Section 3 until Section 2 is approved.


---

## [2026-03-11 18:40] Section 2 — APPROVED. Proceed to Section 3.

Writer, Section 2 is approved. Fixes verified directly from the files (bibliographic entries and draft confirmed correct — I did not rely on your `to_reviewer.md` message, which appears to have been a Phase 1 duplicate appended by accident; please be more careful with copy-paste in future reports).

**What was approved:**
- DocBank (`li2020docbank`) and DocLayNet (`pfitzmann2022doclaynet`) added to `sources/papers_annotated.md` — both entries are clean with correct author lists and venues
- HunyuanOCR superlative corrected to "one of the leading open-weight OCR VLMs" ✓
- GOT-OCR "580M parameters" accepted as stated (consistent with the paper)
- All 21 cited papers have matching BibTeX entries in the bibliography ✓

**Bibliography count: 24 entries.** This is adequate for Phase 2 completion.

---

### Section 3 — Dataset (`draft/sec3_dataset.md`)

You are now cleared to write Section 3. This is the most information-dense section of the paper — describe what we built, not just what we plan to build.

**Target**: 1,000–1,500 words. File: `draft/sec3_dataset.md`.

**Required subsections:**

**3.1 Overview and Design Principles**
State the four target features (strikethrough, underline, color, complex financial tables) and the design decision to produce a unified multi-feature evaluation set. Justify: why financial documents, why these four features, why existing benchmarks are insufficient (cross-reference §2.2 gap argument).

**3.2 Data Sources**
Describe each source pipeline with specifics:
- **SynFinTabs** (Bradley et al., 2024): 5,000 training samples + 200 eval (50/feature). Source: `ethanbradley/synfintabs` on HuggingFace.
- **EDGAR filings**: 1,301 real financial table pairs extracted from SEC EDGAR public filings pipeline. Describe the pipeline briefly (HTML→image rendering, structure extraction).
- Note that OmniDocBench was rate-limited during data collection; SynFinTabs used as fallback for general OCR diversity.

**3.3 Annotation Schema**
Describe the output format for each feature:
- Strikethrough: `~~text~~`
- Underline: `<u>text</u>`
- Color: `<span style="color:NAME;">text</span>`
- Complex tables: HTML with `colspan`/`rowspan` attributes

**3.4 Evaluation Set**
200 pages (50/feature), bootstrap JSONL at `dataset/eval/eval_bootstrap.jsonl`. Annotation protocol: two independent annotators, Cohen's kappa ≥ 0.80 target. Note annotation is pending (present tense for design, future/passive for pending work).

**3.5 Training Set Statistics**
Provide the table from `project_plan.md`:
| Feature | Sources | Pairs | ~Tokens | Status |
Include the totals. Keep honest about what is complete vs. what is planned.

**Sources to draw from:**
- `project_plan.md` — dataset pipeline details and statistics table
- `sources/papers_annotated.md` — SynFinTabs, FinTabNet.c, EDGAR citations

Post to `to_reviewer.md` when complete. Do not begin Section 4 until Section 3 is approved.


---

## [2026-03-11 18:52] Section 3 Review — NEEDS ONE FIX

Writer, I have read `draft/sec3_dataset.md`. The structure and content are strong — the EDGAR pipeline description is detailed and credible, the annotation schema table is clear, and the honest disclosure about annotation being pending is appropriate. One blocking fix required.

---

### Blocking — SynFinTabs 200-sample contradiction

**§3.2** says: *"A 200-sample test subset is **reserved for baseline evaluation** of complex table recognition."*

**§3.5** table shows: *"Complex tables | FinTabNet.c (2,064) + SynFinTabs test (200) + EDGAR (1,301) | 3,565 | Complete"*

These are contradictory. "Reserved for baseline evaluation" implies the 200 samples are withheld from training. But the 3,565 count in §3.5 includes them. The project plan explicitly counts SynFinTabs (200) in the training set (line: `FinTabNet.c (2,064) + SynFinTabs (200) + EDGAR (1,301) | 3,565`).

**Fix**: Replace the §3.2 sentence with:

> A 200-sample SynFinTabs test subset is used for **pre-fine-tuning baseline evaluation** of vanilla HunyuanOCR on complex table recognition; these samples are also included in the complex table training count in §3.5 (no data leakage: the baseline is measured before any fine-tuning).

This makes the dual-use explicit and removes the ambiguity.

---

### Non-blocking (fix in same pass)

**Missing ~Tokens column in §3.5 table.** The project roadmap includes token counts. Add a `~Tokens` column between `Pairs` and `Status`. Values from the project plan:
- Complex tables: ~55K (target; current count yields proportionally less)
- Strikethrough: ~5K
- Underline: ~5K
- Color: ~10K
- Replay: ~5K

---

### What's strong (no changes needed)

- §3.1 motivation is tight and correctly cross-references §2.2 gap argument ✓
- EDGAR pipeline specifics (10-K/A, DEF 14A, S-1/A; 3-row/2-col/merged-cell filter; HTML sanitization) are concrete ✓
- Seven named color vocabulary justified by "closed vocabulary simplifies evaluation to classification" ✓
- Honest acknowledgment that 3,565 complex table pairs fall short of 55K roadmap target ✓
- "Chen et al., EMNLP 2021" = FinQA (Zhiyu Chen) and "Zhu et al., ACL 2021" = TAT-QA (Fengbin Zhu) — both verified in bibliography ✓

---

### Communication note

This is the third time your `to_reviewer.md` report has been a copy-pasted Phase 1 resubmission message rather than an actual Section 3 completion report. I'm reading the files directly so this hasn't blocked review, but please post a proper completion report. Fix the template you are using.

---

Apply the two fixes and post a proper completion report to `to_reviewer.md`. Fast pass — both are one-sentence changes.


---

## [2026-03-11 19:02] Section 3 — APPROVED. Proceed to Section 4.

Writer, Section 3 is approved. Both fixes verified directly from the file — the SynFinTabs clarification is precise and the ~Tokens column is consistent with the rest of the table.

Your `to_reviewer.md` reports are still copy-pasting Phase 1 content. I've now flagged this four times. Fix your template: each new report should have a fresh header, the section number you completed, word count, and any self-assessment notes. That's all.

---

### Section 4 — Method (`draft/sec4_method.md`)

You are cleared to write Section 4. This is the most technical section — be precise, use equations, and keep claims tied to what we actually implement.

**Target**: 1,200–1,800 words. File: `draft/sec4_method.md`.

**Required subsections:**

**4.1 Baseline Model**
Describe HunyuanOCR as our starting point: native-resolution ViT encoder (0.4B), Hunyuan-0.5B LLM decoder, end-to-end RL training. One short paragraph. Forward-reference §3 training data.

**4.2 Dual LoRA Adaptation**
This is the core methodological contribution. Include:
- The full weight update decomposition equation: $\Delta W = \frac{\alpha}{\sqrt{r_1 r_2}} \operatorname{ReLU}(BA) \odot \operatorname{Sign}(DC)$
- Explain the two groups: magnitude group $W_m = \operatorname{ReLU}(BA)$ (selective activation) and direction group $W_d = \operatorname{Sign}(DC)$ (sign binarization)
- The Straight-Through Estimator (STE): gradient flows through Sign as identity during backprop
- Parameter count: 4 matrices (A, B, C, D) vs. 2 in standard LoRA; higher effective rank in $\Delta W$
- Distinguish from DoRA: DoRA decomposes $W_0$ (the pre-trained weight); Dual LoRA decomposes $\Delta W$ (the update) — cite both
- State explicitly: we apply Dual LoRA to both the ViT encoder and the LLM decoder (if this is the plan — confirm from project_plan.md)

**4.3 Multi-Feature Training Objective**
Describe the training setup:
- Single unified model trained jointly on all four features (not four separate models)
- Loss: cross-entropy on the ground truth token sequence (teacher-forced)
- Replay buffer mixing: 10–15% SynFinTabs general samples per batch to mitigate catastrophic forgetting
- If FinTabNet.c is used differently (e.g., for evaluation only vs training), state that clearly

**4.4 Output Format and Decoding**
Explain the output vocabulary extension: the model must generate Markdown strikethrough (`~~...~~`), HTML underline (`<u>...</u>`), HTML color spans (`<span style="color:NAME;">...</span>`), and structural HTML tables with `colspan`/`rowspan`. State whether any special tokens are added or if these are produced via the standard vocabulary.

**Sources:**
- `sources/dual_lora_notes.md` — equations, STE, DoRA comparison
- `sources/papers_annotated.md` — LoRA, DoRA, Dual LoRA BibTeX
- `project_plan.md` — training architecture decisions

Post to `to_reviewer.md` with a proper report when complete. Do not begin Section 5 until Section 4 is approved.


---

## [2026-03-11 19:12] Section 4 Review — NEEDS TWO FIXES

Writer, good report this time — proper header, word count, and design decision noted. Section 4 is technically strong overall but has **two blocking issues** before I can approve.

---

### Blocking fix 1 — Effective rank bound is mathematically incorrect

**Current text (§4.2):** *"The Hadamard product of $\operatorname{ReLU}(BA) \in \mathbb{R}^{d \times k}$ and $\operatorname{Sign}(DC) \in \mathbb{R}^{d \times k}$ can yield an update matrix with rank up to $\min(d, k)^2$"*

**The error**: For two $d \times k$ matrices of ranks $r_1$ and $r_2$, their Hadamard (element-wise) product has rank at most $r_1 \cdot r_2$ — not $\min(d, k)^2$. The upper bound $\min(d,k)^2$ is only achievable if both input matrices are full-rank, which contradicts the low-rank assumption. The correct argument is:

> The Hadamard product of $\operatorname{ReLU}(BA)$ (effective rank $\leq r_1$) and $\operatorname{Sign}(DC)$ (effective rank $\leq r_2$) yields an update matrix with effective rank up to $r_1 \cdot r_2$. For our setting $r_1 = r_2 = r$, this gives up to $r^2$—far exceeding the rank-$r$ constraint of standard LoRA at the same parameter budget.

Replace the current sentence with this corrected version. The argument is still strong ($r^2$ vs $r$ is a meaningful improvement); you just need the correct bound.

---

### Blocking fix 2 — Initialization claim is not in the paper notes and must be reframed

**Current text (§4.2):** *"All four matrices ($A$, $B$, $C$, $D$) are initialized from random Gaussian distributions (not zero), with a warm-up strategy in early training steps to ensure $\Delta W \approx 0$ at initialization."*

I checked `sources/dual_lora_notes.md`. The initialization of Dual LoRA's four matrices is **not documented there**. The notes only record standard LoRA's initialization (B=zero, A=Gaussian). Your reasoning is logically correct (zero-initializing $B$ would block ReLU gradients), but this cannot be attributed to the paper unless you can quote a specific section.

**Fix**: Reframe this as our implementation decision:

> In our implementation, we initialize all four matrices ($A$, $B$, $C$, $D$) from $\mathcal{N}(0, \sigma^2)$ with small $\sigma$ (rather than zero-initializing $B$ and $D$ as in standard LoRA), to avoid blocking gradient flow through the ReLU gate at initialization. A brief linear warm-up over the first 50 steps scales the adapter contribution from 0 to its full value.

If the original paper (arXiv:2512.03402) does describe their initialization scheme, quote it and give the section number. Otherwise use the reframing above.

---

### Non-blocking note

**"1B-parameter" for HunyuanOCR (§4.1)**: 0.4B ViT + 0.5B LLM = 0.9B. "~1B-parameter" is acceptable shorthand if that's how the paper describes itself. If the paper says "0.9B" or "900M", match the paper's wording. Not blocking.

---

### What's strong (confirmed, no changes)

- Dual LoRA equations match `dual_lora_notes.md` exactly ✓
- STE formula is correct ✓
- DoRA vs Dual LoRA distinction ($W_0$ vs $\Delta W$) is precise ✓
- $r_1 = r_2 = r$ matching LoRA rank $2r$ parameter budget — verified correct ✓
- §4.3 FinTabNet.c as eval-only is consistent with §3 ✓
- §4.4 no-special-tokens, closed 7-color vocab, single decoding pass ✓
- ViT frozen / LLM-only primary config with §5 ablation noted ✓

---

Apply both fixes and post a proper completion report to `to_reviewer.md`.


---

## [2026-03-11 19:22] Section 4 — APPROVED. Proceed to Section 5.

Writer, Section 4 is approved. Both fixes are correct:
- Rank bound corrected to $r_1 \cdot r_2$ (mathematically sound) ✓
- Initialization sourced directly from Xu et al. §3.3 with block quote ✓
- Good instinct to update `dual_lora_notes.md` with the initialization section — that's the right way to handle new findings from the paper.

---

### Section 5 — Experiments (`draft/sec5_experiments.md`)

You are cleared to write Section 5. **This section uses placeholders** — results are not yet available. Write a complete skeleton with all structure, metrics, baselines, and ablations described, inserting `[TBD]` where actual numbers will go.

**Target**: 1,000–1,400 words. File: `draft/sec5_experiments.md`.

**Required subsections:**

**5.1 Experimental Setup**
- Hardware: describe training hardware (check `project_plan.md` for any specified config; if not specified, write "NVIDIA A100 80GB GPUs [TBD: count]")
- Optimizer: AdamW, learning rate, batch size — use project_plan.md values if available, else [TBD]
- Dual LoRA rank: $r_1 = r_2 = r$; state the value of $r$ used (from project_plan.md or [TBD])
- Training epochs: [TBD]
- Evaluation: held-out 200-page set (§3.4); TEDS for tables (Zhong et al., ECCV 2020); feature-specific metrics for formatting features (exact match on format token sequence)

**5.2 Main Results**
Present a results table with placeholder values. Rows = models, Columns = per-feature metrics + overall:

| Model | Strikethrough F1 | Underline F1 | Color Acc. | Table TEDS | Overall |
|---|---|---|---|---|---|
| HunyuanOCR (vanilla) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + Standard LoRA | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| + Dual LoRA (ours) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| Tan et al. (2025)† | — | — | — | [TBD] | — |

Include a footnote explaining the Tan et al. row only covers tables (†: single-feature, different dataset).

**5.3 Ablation Study**
Two ablations from the project plan:

*Ablation 1 — ViT encoder tuning*: Compare (a) LLM-only Dual LoRA (primary config) vs. (b) LLM+ViT Dual LoRA. Table with same metrics. Hypothesis: ViT tuning helps strikethrough/underline/color (visual features) but risks forgetting layout understanding.

*Ablation 2 — Replay buffer*: Compare with vs. without the SynFinTabs replay buffer. Metric: CER/WER degradation on OmniDocBench general OCR subset. Hypothesis: without replay, CER/WER degrades ≥5%.

**5.4 Error Analysis** (one paragraph, qualitative)
Describe expected failure modes based on dataset characteristics — e.g., short strikethrough spans confused with hyphens, color names for low-saturation colors, merged-cell depth > 3 levels. Write this as "we expect" / "preliminary inspection suggests" since results are pending.

**Sources:**
- `project_plan.md` — hardware, training config, ablation designs
- `draft/sec4_method.md` — metric definitions, baseline descriptions
- `sources/papers_annotated.md` — Tan et al. citation

Post to `to_reviewer.md` with a proper report when complete. Do not begin Section 1 (Introduction + Abstract) until Section 5 is approved.


---

## [2026-03-11 19:37] Follow-up — Section 5 Not Started

Writer, `draft/sec5_experiments.md` has not been created and `to_reviewer.md` is unchanged since your Section 4 report (~15 min ago).

Your task is `draft/sec5_experiments.md` — a skeleton with `[TBD]` placeholders, no actual results required. The brief is in this file under **"Section 5 — Experiments"**. This should be a fast write.

If your session restarted, re-read the Section 5 brief and begin immediately. Post an acknowledgment to `to_reviewer.md` if you are running.

