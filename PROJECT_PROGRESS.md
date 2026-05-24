# Project Progress Log

## Project Overview
- Project: SFEARNet-based remote sensing change detection research and paper writing.
- Code root: `/home/fengruyue/lvguixin/SFEARNet_test`
- Manuscript root: `/home/fengruyue/lvguixin/SFEARNet_test/manuscript`
- Main challenge: pseudo-change (visual change exists but dataset labels it as unchanged), and improving prediction consistency with benchmark annotations.

## Confirmed Innovation Points (from user + code inspection)
1. `loss/ContrastiveLoss.py` -> `ContrastiveLoss`:
   - Adds a feature-level contrastive supervision between bi-temporal features (`feat1`, `feat2`).
   - Directly applies channel-wise L2 normalization to the two temporal deep features.
   - Computes per-pixel Euclidean distance and contrastive objective conditioned on GT change labels.
   - Loss is normalized and clamped to `[0,1]` for stable joint optimization.
2. `utils/stages.py` lines ~66-79:
   - Constructs a three-branch weighted objective:
     - Segmentation branch: CE + IoU (with internal CE normalization and IoU clamp)
     - Contrastive branch
     - Edge branch
   - Uses coefficients (`alpha`, `beta`, `gamma`) and convex-like fusion:
     - `loss_seg = alpha * CE_norm + (1-alpha) * IoU_norm`
     - `loss = (1-beta-gamma) * loss_seg + beta * contrast + gamma * edge`
     - final clamp to `[0,1]`

## Current Understanding (assistant interpretation)
- Base architecture stays as SFEARNet (semantic flow + edge-aware refinement pipeline).
- Your contribution is mainly training-objective level rather than backbone-level:
  - introducing feature discrimination constraints aligned with label semantics,
  - and rebalancing multi-task losses through bounded normalization,
  - to suppress predictions on pseudo-change-like regions that are visually changed but label-unchanged.
- In effect, you are pushing model behavior closer to benchmark annotation policy (dataset-defined change), not purely photometric difference detection.

## Method Extraction From Current Code
### 1. Baseline vs. New Part
- Baseline part:
  - The backbone and prediction pipeline are inherited from `SFEARNet`, which outputs:
    - change prediction `label_pred`,
    - edge prediction `edge_pred`,
    - bi-temporal deep features `feat1, feat2`.
- New part:
  - A pixel-wise feature contrastive loss is introduced in `loss/ContrastiveLoss.py`.
  - A bounded three-branch multi-task joint optimization strategy is defined in `utils/stages.py`.

### 2. Algorithm Flow
1. Input a bi-temporal image pair `(I_A, I_B)` into the network.
2. The network outputs:
   - semantic change logits `P`,
   - edge logits `E`,
   - deep features `F_A, F_B`.
3. For contrastive supervision:
   - `F_A, F_B` are directly L2-normalized along the channel dimension.
   - A per-pixel Euclidean distance map is computed.
4. The distance map is supervised by the binary change label:
   - unchanged pixels: minimize feature distance;
   - changed pixels: enforce a margin-separated distance.
5. The final training objective fuses:
   - segmentation supervision,
   - edge supervision,
   - contrastive supervision.
6. Each branch is normalized/clamped to a bounded interval before weighted fusion, so the final objective remains in `[0,1]`.

### 3. Objective Function (implementation-consistent)
Let:
- `y_ij in {0,1}` be the ground-truth change label at pixel `(i,j)`, where `0` means unchanged and `1` means changed.
- `f_A(i,j), f_B(i,j)` be the directly L2-normalized deep features at pixel `(i,j)`.
- `d_ij = ||f_A(i,j) - f_B(i,j)||_2` be the per-pixel feature distance.
- `m` be the contrastive margin.

Direct feature normalization:

`f_A = Norm(F_A), f_B = Norm(F_B)`

where `Norm(.)` denotes channel-wise L2 normalization.

Pixel-wise contrastive loss:

`L_ctr(i,j) = (1 - y_ij) * 0.5 * d_ij^2 + y_ij * 0.5 * max(0, m - d_ij)^2`

Image-level contrastive loss in code:

`L_ctr = clamp(mean(L_ctr(i,j)) / max(1, 0.5*m^2), 0, 1)`

Segmentation branch normalization:

`L_ce_norm = 1 - exp(-L_ce)`

`L_iou_norm = clamp(L_iou, 0, 1)`

Segmentation fusion:

`L_seg = alpha * L_ce_norm + (1 - alpha) * L_iou_norm`

Edge branch normalization:

`L_edge_norm = clamp(L_edge, 0, 1)`

Final multi-task joint optimization objective:

`L_all = (1 - beta - gamma) * L_seg + beta * L_ctr + gamma * L_edge_norm`

and the code further uses:

`L = clamp(L_all, 0, 1)`

Coefficient constraints from code:

`0 <= alpha <= 1, 0 <= beta <= 1, 0 <= gamma <= 1, beta + gamma <= 1`

Current training setup from `train.py`:
- `margin = 3.0`
- default args: `alpha = 0.5`, `beta = 0.1`, `gamma = 0.4`

### 4. Innovation Points (paper-ready interpretation)
1. Feature-level label-guided contrastive supervision:
   - Unlike pure pixel classification, the method explicitly constrains bi-temporal deep features according to benchmark change semantics.
   - This helps reduce pseudo-change responses by pulling unchanged-region features closer and pushing changed-region features apart up to a margin.
2. Bounded multi-task joint optimization:
   - CE, IoU, edge, and contrastive objectives are normalized to a comparable numeric scale before fusion.
   - This avoids one branch dominating optimization because of raw magnitude mismatch.
3. Annotation-consistent change modeling under pseudo-change disturbance:
   - The method is designed to fit the dataset definition of change, not merely visual difference intensity.
   - This is the most important narrative link between your code design and the remote-sensing problem setting.

### 5. Caution on Formula Description
- The current code uses the standard contrastive assignment:
  - unchanged label `0`: minimize distance;
  - changed label `1`: enforce margin separation.
- If writing the paper, the text and symbols must match this exact convention, otherwise it will conflict with implementation.
- Because features are L2-normalized, the theoretical Euclidean distance upper bound is `2`; however, the current implementation sets `margin = 3.0`.
- This does not break code execution because loss is normalized and clamped, but it is worth noting in later experiment discussion or margin ablation.

## Evidence Locations
- `loss/ContrastiveLoss.py`
- `utils/stages.py`
- `train.py` (initializes `ContrastiveLoss(in_channels=model.in_channels[3], margin=3.0)` and passes weighted losses)

## Notes / Potential Follow-up
- `ContrastiveLoss` currently uses label directly as positive/negative selector for changed/unchanged; this matches benchmark-defined supervision.
- Margin is currently set to `3.0` in `train.py` and internally normalized by `max(1.0, 0.5*margin^2)`.
- If paper writing continues, recommend explicitly framing this as “annotation-consistent change modeling under pseudo-change disturbance.”

## Work Log
### 2026-05-05
- Read and analyzed target code blocks specified by user.
- Traced integration in training pipeline (`train.py` + `utils/stages.py`).
- Created persistent handover files: `PROJECT_PROGRESS.md`, `HANDOVER_PROMPT.txt`.
- Produced initial technical interpretation for user confirmation.

## Handover Rule
- Keep updating this file after each substantial discussion/code change.
- New session assistant should read this file first, then inspect referenced source files before making claims.

### 2026-05-05 (Update 2)
- User requested a Chapter 3 structure aligned with SFEARNet Section III (Methodology) and strict LaTeX indentation style.
- Prepared a methodology-structured outline template with IEEE section/subsection hierarchy for direct use in manuscript drafting.
- Waiting for user-provided per-subsection content bullets to expand the outline into full Chapter 3 text.

### 2026-05-05 (Update 3)
- User specified Chapter 3 first subsection should focus on limitations of existing methods and pseudo-change introduction.
- Drafted a formal LaTeX subsection text (problem motivation + pseudo-change definition + impact) for direct insertion.

### 2026-05-05 (Update 4)
- User requested terminology correction for research summary.
- Updated framing rules:
  1) Do not mention SFEARNet in research-language summary.
  2) Replace "multi-task loss" wording with "multi-task joint optimization".
  3) In subsequent LaTeX writing, explicitly present the multi-task joint optimization objective function, instead of vague "multi-loss" wording.

### 2026-05-05 (Update 5)
- User requested innovation summary to be decoupled into 3-4 points.
- Constraint: 2-3 points for method ideas/thoughts + 1 point for experimental contribution.

### 2026-05-07 (Recovery Update)
- Context recovery completed after accidental interruption.
- Re-checked code-level evidence in:
  - `loss/ContrastiveLoss.py`
  - `utils/stages.py`
  - `train.py`
- Recovered consolidated conclusion (for direct continuation):
  1) Baseline part (existing): encoder-decoder flow and edge-aware pipeline inherited from SFEARNet family design.
  2) New method part (current project): feature-level contrastive constraint + bounded multi-task joint optimization.
  3) Optimization objective (implementation-consistent):
     - `loss_seg = alpha * CE_norm + (1-alpha) * IoU_norm`
     - `loss_all = (1-beta-gamma) * loss_seg + beta * contrast + gamma * edge`
     - all branches and final loss are clamped/normalized to `[0,1]`.
  4) Research framing: target is annotation-consistent change detection under pseudo-change disturbance, rather than only visual-difference sensitivity.
- Writing constraints recovered and kept active:
  - In research-style summary, avoid directly using baseline model name as innovation claim.
  - Use term: "multi-task joint optimization" (not vague "multi-loss").
  - Explicitly present objective equations in Method section.

### 2026-05-07 (Maintenance Update)
- `PROJECT_PROGRESS.md` is now resumed as the single source of truth for session continuity.
- `sfearnet_ref.txt` will be maintained as mixed content:
  - paper/source excerpts,
  - structured project notes for writing and experiments.
- Next step queue (paper + experiment, executable):
  1) Method writing: finalize Chapter 3 subsection text with explicit symbols for `alpha, beta, gamma, margin`.
  2) Ablation plan: run/organize `beta,gamma,margin` sensitivity and report IoU/F1/OA/Kappa.
  3) Reproducibility: ensure logs/csv/model paths are cited in manuscript experiment section.

### 2026-05-07 (Update 8)
- User requested generating full Chapter 3 content based on:
  - confirmed innovation points (4 items),
  - confirmed research summary,
  - previous Chapter 3 formatting constraints.
- Replaced `manuscript/Sections/4-Methodology.tex` with complete methodology text.
- New Chapter 3 includes:
  1) problem definition and pseudo-change motivation;
  2) overall framework description (three-branch supervision);
  3) feature-level contrastive supervision with margin-based formula;
  4) bounded multi-task joint optimization with explicit `alpha/beta/gamma` constraints and normalization;
  5) direct mapping from method design to five-part experiment protocol.
- Status: Chapter 3 draft is now implementation-consistent with current code (`ContrastiveLoss`, `stages`, `train`).
- Next step: draft `Sections/5-Experiment.tex` to align exactly with the five experiments and six metrics.

### 2026-05-07 (Update 9)
- Re-read and extracted the exact implementation from:
  - `loss/ContrastiveLoss.py`
  - `utils/stages.py`

### 2026-05-11 (Update 10)
- User requested a flowchart plan for Chapter 4, Section 2 (`Overview`).
- Re-checked the current methodology text in `manuscript/Sections/4-Methodology.tex` and the implementation in `model/SFEARNet.py`, `loss/ContrastiveLoss.py`, `utils/stages.py`, and `utils/args_utils.py`.
- Confirmed the diagram should be a left-to-right pipeline centered on the shared bi-temporal network output:
  1. bi-temporal inputs;
  2. shared SFEARNet forward pass;
  3. three training branches:
     - segmentation (CE + IoU);
     - edge supervision;
     - label-guided contrastive supervision;
  4. bounded normalization and weighted fusion;
  5. final clipped objective and backpropagation.
- Writing guidance for the paper:
  - keep the figure at the overview level, not equation-detail level;
  - show the contrastive branch and bounded fusion as the two novel parts;
  - leave CE/IoU/edge formula details to the later subsections.

### 2026-05-11 (Update 11)
- User requested a formatting-only revision of the Related Work section.
- Main change requested:
  - move each `\cite{}` closer to the specific sentence that introduces the corresponding paper;
  - avoid grouping multiple papers under a single end-of-sentence citation when the text is describing them one by one.
- Applied rewrite to `manuscript/Sections/2-Related_Work.tex`:
  - split combined citation blocks into per-paper citations where appropriate;
  - kept the existing content structure and conclusions unchanged;
  - improved TGRS-style readability by aligning each author contribution with its immediate citation.

### 2026-05-11 (Update 12)
- Clarified the IEEE/TGRS citation-placement rule for related work:
  - the preferred default is to place the citation immediately after the author name or after the exact clause being supported;
  - avoid leaving a citation block at the end of a sentence when it covers multiple distinct papers with different contributions.
- Current writing rule for this manuscript:
  - use per-paper citations placed as close as possible to the corresponding contribution sentence;
  - keep the narrative point-to-point and avoid citation bundling across separate works.

### 2026-05-11 (Update 13)
- Unified the entire `manuscript/Sections/2-Related_Work.tex` section to the same citation style:
  - `Author \cite{...} proposed ...`
  - `Author \cite{...} introduced ...`
  - `Author \cite{...} developed ...`
- Result:
  - each cited paper now has its own immediately adjacent citation marker;
  - multi-paper grouping at the end of one sentence has been removed where it reduced readability;
  - the section now follows a more standard IEEE/TGRS related-work rhythm.

### 2026-05-11 (Update 14)
- User requested a factual audit of the Related Work summaries.
- Verification outcome after checking titles/abstracts from publisher or institutional pages:
  - the current section is broadly faithful to the cited papers;
  - no obvious fabricated paper contribution was found;
  - most sentences are safe as literature-summary paraphrases.
- Caveats identified:
  1) `CDasXORNet` is most directly a building change-detection method; describing it as pseudo-change-related is acceptable only as a broader inference about building-change ambiguity.
  2) `RSSFormer` is a land-cover segmentation paper, so it is appropriate only as a dense-prediction design reference, not as a change-detection contribution.
  3) Some closing synthesis sentences in the section are author-side generalizations, not claims made verbatim by individual papers; they should be read as review-level conclusions.
- Overall recommendation:
  - keep the current content structure;
  - retain the stricter citation placement;
  - if even tighter factual tone is desired, soften the two caveat cases above.

### 2026-05-11 (Update 15)
- Applied the requested factual-tone refinements to `manuscript/Sections/2-Related_Work.tex`.
- Main wording adjustments:
  - `SASiamNet` rewritten as a self-adaptive Siamese framework, which is closer to the paper title and avoids over-specific phrasing.
  - `CDasXORNet` rewritten as an XOR-style binary relation problem, which is a safer way to describe the building-change formulation.
  - `RSSFormer` explicitly marked as a semantic-segmentation reference rather than a change-detection contribution.
- Result:
  - the section is now more conservative and less likely to overstate what each paper actually claims.

### 2026-05-11 (Update 16)
- User requested running and recording the baseline comparison methods stored under `baselines/`.
- Current code inventory:
  - 9 code directories cover 10 change-detection baselines.
  - `fully_convolutional_change_detection` covers `FC-EF`, `FC-Siam-Conc`, and `FC-Siam-Diff`.
  - Other directories correspond to `STANet`, `BIT`, `ChangeFormer`, `DESSN`, `AMTNet`, `USSFC`, and `EATDer`.
- Local environment status:
  - GPU is available (`RTX 4090` detected).
  - Local datasets exist for `LEVIR_CD_256`, `WHU-CD-256`, and `CLCD_256`.
  - `GZ-CD` is not fully present in the workspace.
  - No baseline checkpoints were found under the baseline folders.
- Practical implication:
  - full reproduction is blocked until official pretrained weights are downloaded or each model is trained from scratch;
  - evaluation scripts alone are not enough to produce baseline metrics without checkpoints.

### 2026-05-13 (Update 17)
- User requested a presentation-oriented method summary before running baseline code.
- Created a new file: `汇报.md` at repo root.
- Content scope:
  - problem motivation (pseudo-change and annotation consistency);
  - method overview;
  - two core innovations (label-guided contrastive supervision + bounded multi-task joint optimization);
  - key formulas aligned with current methodology text;
  - short oral script structure for a 5-8 minute report.

### 2026-05-14 (Update 18)
- User reported formula overlap in Chapter 3 screenshot.
- Updated `manuscript/Sections/3-Preliminary.tex` to improve equation layout in `Common Loss Functions`:
  - kept CE loss as an independent equation;
  - split IoU and Edge definitions into compact main equations plus separate component equations;
  - added dedicated equation labels for components: `eq:iou_components`, `eq:edge_components`;
  - adjusted wording from repeated `where` to `Here` for cleaner narrative flow.
- Expected result:
  - right-side equation number crowding is reduced;
  - no three-formula packing in one visual block;
  - improved IEEE two-column readability.
  - `train.py`
- Added a code-consistent method summary to this file:
  - baseline vs. new contribution boundary,
  - algorithm flow,
  - mathematical objective definitions,
  - innovation-point interpretation,
  - implementation caveat on `margin = 3.0` under L2-normalized features.
- User's requested persistent handover maintenance continues:
  - `PROJECT_PROGRESS.md` stores status, conclusions, and next-step queue;
  - `HANDOVER_PROMPT.txt` stores the next-session prompt contract.
- Immediate next recommended tasks:
  1) convert the formulas here into manuscript notation style;
  2) verify whether `margin=3.0` is intentional or should be constrained/ablated;
  3) write Experiment section around `alpha/beta/gamma/margin` sensitivity and pseudo-change robustness.

### 2026-05-07 (Update 10)
- User requested generating an algorithm description from the confirmed innovation code and innovation-point summary.
- Output direction fixed for subsequent writing:
  - generate a paper-ready method algorithm, not just prose summary;
  - keep it consistent with current code flow:
    1) dual-temporal input;
    2) network outputs segmentation, edge, and deep features;
    3) projected normalized feature distance computation;
    4) label-guided contrastive supervision;
    5) bounded three-branch joint optimization.
- Recommended immediate reuse:
  - provide one Chinese algorithm description paragraph;
  - provide one pseudocode / Algorithm block version for manuscript insertion.

### 2026-05-07 (Update 11)
- User asked whether the method can be split into two algorithms.
- Confirmed recommendation:
  1) Algorithm 1: label-guided pixel-wise feature contrastive constraint;
  2) Algorithm 2: bounded three-branch multi-task joint optimization and training procedure.
- Reason for split:
  - Algorithm 1 highlights the core feature-space innovation.
  - Algorithm 2 highlights the objective-level fusion and end-to-end optimization process.
- This split is more suitable for the Method section because it separates "feature supervision mechanism" from "overall optimization pipeline".

### 2026-05-07 (Update 12)
- User asked whether these two algorithms are sufficient to support publishing a paper.
- Current assessment:
  - Two algorithms are structurally enough for the Method section if the paper's main contribution is training-objective innovation rather than backbone redesign.
  - But publication sufficiency does not depend on "number of algorithms"; it depends on whether the full paper proves:
    1) a clear problem motivation,
    2) a nontrivial method contribution,
    3) implementation-consistent mathematical formulation,
    4) convincing experiments and ablations,
    5) visible advantage over baselines.
- For this project, the strongest publishable framing remains:
  - pseudo-change suppression / annotation-consistent change detection;
  - feature-level contrastive supervision;
  - bounded multi-task joint optimization.
- Immediate paper-readiness gap is more likely in experiments and evidence, not in whether the method is split into one or two algorithms.

### 2026-05-07 (Update 13)
- User selected the next task: assess what publication level this method is more suitable for.
- Current recommendation:
  - Position the work as an application-oriented or improvement-oriented remote-sensing change detection paper.
  - Do not overclaim it as a brand-new backbone/network architecture paper, because the major novelty is in supervision and optimization rather than encoder-decoder structure.
- Practical target judgment:
  1) More suitable:
     - general remote sensing / image analysis journals or conferences that accept method enhancement on a known baseline;
     - venues where problem framing + ablation + application value matter more than radical architectural novelty.
  2) Less suitable:
     - top-tier method venues that expect a substantially new architecture, stronger theory, or a large performance gap across multiple benchmarks.
- Deciding factor remains experimental strength:
  - if pseudo-change analysis, ablation, and benchmark gains are solid, the method can support a respectable improvement paper;
  - if gains are small or only shown on limited settings, target level should be lowered accordingly.

### 2026-05-07 (Update 14)
- User selected the next task: design an experiment scheme that can support publication and assess current evidence gaps.
- Repository status checked:
  - dataset experiment scripts already exist for `LEVIR_CD_256`, `WHU-CD`, and `CLCD`;
  - training currently reports `OA`, `PA`, `IoU`, `Recall`, `F1`, `Kappa` and writes csv logs;
  - `alpha`, `beta`, `gamma` are configurable via args, but `margin` is still hard-coded in `train.py`.
- Recommended publication-supporting experiment structure:
  1) benchmark comparison across multiple datasets;
  2) core ablation for contrastive branch and bounded joint optimization;
  3) hyperparameter sensitivity for `alpha/beta/gamma/margin`;
  4) pseudo-change visualization / case analysis;
  5) efficiency or complexity report if space permits.
- Current main gap is experimental evidence rather than method description.
- Important implementation follow-up:
  - expose `margin` as a runtime argument before formal ablation;
  - ensure each experiment has reproducible script, log path, and model path for manuscript citation.

### 2026-05-07 (Update 15)
- User requested generating the mathematical formulas corresponding to the two split algorithms.
- Output requirement fixed:
  1) Algorithm 1 should correspond to the feature projection, normalization, distance computation, and label-guided contrastive loss;
  2) Algorithm 2 should correspond to segmentation/edge/contrastive branch normalization and bounded multi-task joint optimization.
- Reuse rule for later writing:
  - keep formula notation directly aligned with current code convention:
    - `y_ij=0` unchanged,
    - `y_ij=1` changed,
    - `beta+gamma<=1`,
    - final objective clamped to `[0,1]`.

### 2026-05-07 (Update 16)
- User requested writing the two algorithms and their corresponding formulas into a dedicated `recode.txt` file.
- Created new file:
  - `recode.txt`
- File content includes:
  1) Algorithm 1 description, steps, formulas, and interpretation;
  2) Algorithm 2 description, steps, formulas, and interpretation;
  3) paper-mapping notes and implementation-consistency notes.
- This file can now be reused directly as:
  - a handover note,
  - a paper drafting source,
  - a formula reference for future LaTeX conversion.

### 2026-05-08 (Update 17)
- User started preparing a PPT for reporting:
  - motivation,
  - innovation points,
  - algorithm flow,
  - corresponding formulas.
- Guidance direction fixed:
  - the slide logic should start from the pseudo-change problem, not from loss definitions;
  - then separate baseline framework, innovation modules, algorithm flow, and formulas;
  - the narrative should highlight "annotation-consistent change detection" as the central theme.
- Recommended PPT backbone:
  1) background and task;
  2) problem motivation;
  3) limitations of existing methods;
  4) overall idea of the proposed method;
  5) innovation point 1 + algorithm 1 + formulas;
  6) innovation point 2 + algorithm 2 + formulas;
  7) expected effect / experiment design / summary.

### 2026-05-08 (Update 18)
- User asked for a concrete writing suggestion for the first PPT page.
- Recommendation fixed:
  - Page 1 should be a background-introduction slide, not a method slide.
  - Its purpose is to establish:
    1) what remote sensing change detection is,
    2) why it matters,
    3) why this task is worth studying.
- Writing strategy:
  - keep it broad and application-oriented;
  - do not introduce formulas or innovation points yet;
  - reserve pseudo-change and method details for later slides.

### 2026-05-08 (Update 19)
- User asked to proceed to the next PPT page after the background slide.
- Next page recommendation:
  - Page 2 should be "Problem Statement / Motivation".
  - It should transition from general application value to the concrete pain point of pseudo-change.
- Writing goal:
  - explain why visual difference is not equal to semantic change;
  - show that existing methods tend to over-detect pseudo-change;
  - naturally motivate the proposed label-consistent contrastive and bounded optimization design.

### 2026-05-08 (Update 20)
- User requested the content for PPT page 3.
- Page 3 should explain the limitations of existing methods.
- Writing goal:
  - transform the pseudo-change problem into a gap in current methods;
  - emphasize that prior approaches often focus on pixel-wise discrepancy or direct prediction, but lack explicit label-consistent feature constraint;
  - create a natural transition to the proposed feature contrastive supervision and bounded joint optimization.

### 2026-05-08 (Update 21)
- User requested the content for PPT page 4.
- Page 4 should present the overall idea of the proposed method.
- Writing goal:
  - summarize the method in one slide before entering algorithm details;
  - make clear that the proposed framework has two core innovations:
    1) label-guided feature contrastive supervision;
    2) bounded multi-task joint optimization;
  - prepare the audience for the split presentation of Algorithm 1 and Algorithm 2.

### 2026-05-08 (Update 22)
- User requested the content for PPT page 5.
- Page 5 should introduce Innovation 1 / Algorithm 1 at the idea level.
- Writing goal:
  - explain why feature-level contrastive supervision is needed;
  - show the mechanism "unchanged close / changed apart";
  - reserve dense formula details for the next slide if needed.

### 2026-05-08 (Update 23)
- User requested the concrete flow of Algorithm 1 for PPT presentation.
- Recommended presentation form:
  - describe Algorithm 1 as a six-step feature-space supervision process;
  - emphasize the sequence:
    1) feature extraction,
    2) projection,
    3) normalization,
    4) distance computation,
    5) label-guided contrastive constraint,
    6) normalized contrastive loss output.
- This flow should be used as the bridge between the idea slide of Innovation 1 and the formula slide of Algorithm 1.

### 2026-05-08 (Update 24)
- User requested the formulas corresponding to Algorithm 1 for PPT use.
- Output direction fixed:
  - present Algorithm 1 formulas in the order of:
    1) feature projection,
    2) feature normalization,
    3) pixel-wise distance,
    4) label-guided contrastive loss,
    5) image-level normalized contrastive objective.
- Presentation requirement:
  - keep formulas short enough for one slide;
  - explicitly state `y_ij=0` unchanged and `y_ij=1` changed.

### 2026-05-08 (Update 25)
- User requested the content for PPT page 7.
- Based on the current slide sequence:
  - Page 5: Innovation 1 / Algorithm 1 idea
  - Page 6: Algorithm 1 formulas
  - Page 7 should move to Innovation 2 / Algorithm 2 idea
- Writing goal:
  - explain why bounded multi-task joint optimization is needed;
  - highlight the three supervision branches:
    1) segmentation,
    2) edge,
    3) contrastive;
  - stress that branch normalization is used to improve optimization stability before weighted fusion.

### 2026-05-08 (Update 26)
- User requested the concrete flow of Algorithm 2 for PPT presentation.
- Recommended presentation form:
  - describe Algorithm 2 as an eight-step joint optimization process;
  - emphasize the sequence:
    1) dual-temporal input,
    2) multi-branch output,
    3) segmentation loss computation,
    4) edge loss computation,
    5) contrastive loss computation,
    6) branch normalization,
    7) weighted fusion,
    8) bounded final loss and parameter update.
- This flow should be used as the bridge between the idea slide of Innovation 2 and the formula slide of Algorithm 2.

### 2026-05-08 (Update 27)
- User requested the content for PPT page 8.
- Based on the current slide sequence:
  - Page 7 introduces Innovation 2 / Algorithm 2 idea
  - Page 8 should present the formulas of Algorithm 2
- Writing goal:
  - show how segmentation, edge, and contrastive losses are normalized and fused;
  - explicitly present the roles of `alpha`, `beta`, and `gamma`;
  - stress that the final objective is bounded to `[0,1]`.

### 2026-05-08 (Update 28)
- User shifted focus from PPT preparation to paper writing.
- Current manuscript status re-checked:
  - `manuscript/Sections/4-Methodology.tex` is already substantially drafted and aligned with current code.
  - `manuscript/Sections/1-Introduction.tex` is still an IEEE template placeholder.
  - `manuscript/Sections/5-Experiment.tex` currently contains only a section skeleton and TODO-style notes.
- Recommended writing order from highest leverage:
  1) finish `Introduction`,
  2) complete `Experiment` section framework,
  3) refine `Related Work`,
  4) polish abstract and conclusion after results are stable.
- Practical note:
  - the shortest path to a coherent manuscript is to use the already-set PPT logic as the paragraph logic for Introduction and Method transition.

### 2026-05-08 (Update 29)
- Re-checked whether previous work is sufficiently clear for handover continuation.
- Conclusion:
  - Yes, the previous work is mostly clear at the project-intent and method-definition level.
  - The clearest sources are `PROJECT_PROGRESS.md` and `HANDOVER_PROMPT.txt`, and both are strongly consistent with:
    - `loss/ContrastiveLoss.py`
    - `utils/stages.py`
    - `train.py`
- What is already clear:
  1) research task is remote sensing change detection under pseudo-change disturbance;
  2) baseline vs innovation boundary is explicitly defined;
  3) the two confirmed innovations are stable and code-supported:
     - label-guided pixel-wise contrastive supervision,
     - bounded multi-task joint optimization;
  4) the current objective formulas and coefficient constraints are already recoverable from code;
  5) the current writing priority has been narrowed to manuscript completion, especially Introduction and Experiment.
- What is still not fully clear / still fragile:
  1) `train.py` fixes `margin=3.0`, while L2-normalized feature distance is theoretically bounded near `2`, so this needs explicit justification or later correction in paper/ablation;
  2) several code files remain modified or newly added, but not all of them are reflected in a concise change log with rationale;
  3) Experiment section is still structurally incomplete compared with the mature Method section;
  4) reproducibility is improving, but dataset-specific scripts and final experiment protocol are not yet summarized into one compact execution guide.
- Recommended next step:
  - treat the project as "method understanding is clear, experiment-and-manuscript closure is not yet complete";
  - next highest-value work should be finishing `Introduction` and `Sections/5-Experiment.tex`, while deciding how to explain or ablate the `margin` setting.

### 2026-05-08 (Update 30)
- User requested inserting the two core algorithms and their corresponding formulas into `manuscript/Sections/4-Methodology.tex` with valid LaTeX formatting.
- Completed edits in Methodology:
  1) expanded the feature-level contrastive subsection with:
     - projection formula,
     - L2 normalization formula,
     - pixel-wise distance formula,
     - pixel-wise contrastive loss,
     - image-level averaging,
     - implementation-consistent normalized/clipped contrastive objective;
  2) inserted `Algorithm 1` for label-guided pixel-wise feature contrastive supervision;
  3) expanded the bounded multi-task joint optimization subsection with:
     - CE normalization,
     - IoU clipping,
     - edge clipping,
     - weighted fusion,
     - coefficient constraints,
     - final clipped objective;
  4) inserted `Algorithm 2` for bounded multi-task joint optimization.
- Code-paper consistency kept aligned with:
  - `loss/ContrastiveLoss.py`
  - `utils/stages.py`
  - `train.py`
- Verification status:
  - text-level LaTeX structure was re-checked after editing;
  - full PDF compilation was not completed because `latexmk` is not installed in the current environment.
- Next recommended step:
  - compile the manuscript in a TeX-enabled environment and adjust float placement if the two algorithm environments affect page layout.

### 2026-05-08 (Update 31)
- User requested moving the definition of pseudo-change from the Methodology chapter to the Preliminary chapter.
- Completed structural adjustment:
  1) replaced the old template-style `manuscript/Sections/3-Preliminary.tex` with project-relevant content;
  2) added three subsections in Preliminary:
     - remote sensing change detection setting,
     - definition of pseudo-change,
     - motivation for annotation-consistent learning;
  3) moved the explicit pseudo-change definition and explanation into Preliminary;
  4) simplified the opening paragraph of `manuscript/Sections/4-Methodology.tex` so it now references Section `Preliminary` instead of redefining pseudo-change in place.
- Result:
  - chapter roles are cleaner now:
    - Preliminary explains concepts and problem background;
    - Methodology focuses on framework, algorithms, and optimization.

### 2026-05-08 (Update 32)
- User requested rewriting the Methodology chapter in a higher-level journal style.
- Reworked `manuscript/Sections/4-Methodology.tex` under the confirmed four-subsection structure:
  1) limitations of existing methods under pseudo-change;
  2) annotation-consistency-oriented design and innovation logic;
  3) Algorithm 1 with formal narrative and formulas;
  4) Algorithm 2 with formal narrative and formulas.
- Main improvements:
  - strengthened problem-to-method logical transition;
  - replaced explanatory/tool-style wording with more formal journal-style academic prose;
  - unified terminology around:
    - pseudo-change,
    - annotation-consistent change detection,
    - feature-level contrastive supervision,
    - bounded multi-task joint optimization;
  - clarified the role of each loss branch and coefficient in a more publishable style.
- Current status:
  - Methodology chapter is now substantially closer to a high-level journal presentation, while still remaining implementation-consistent with current code.

### 2026-05-09 (Update 33)
- User requested continuation of the persistent progress file and handover prompt.
- Re-checked current repository state and refreshed handover context.
- Current confirmed code state:
  1) `model/SFEARNet.py` now returns four outputs:
     - change logits `x`,
     - edge logits `edge`,
     - upsampled deep feature `feat1`,
     - upsampled deep feature `feat2`.
  2) `train.py` initializes `ContrastiveLoss(in_channels=model.in_channels[3], margin=3.0)` and passes it into training/validation.
  3) `utils/args_utils.py` exposes `alpha`, `beta`, and `gamma`; `margin` is still not exposed as a command-line argument.
  4) `utils/stages.py` uses bounded three-branch fusion:
     - `loss_seg = alpha * CE_norm + (1-alpha) * IoU_norm`
     - `loss = (1-beta-gamma) * loss_seg + beta * contrast + gamma * edge`
     - final loss is clamped to `[0,1]`.
  5) Experiment shell scripts now exist for:
     - `experiment_scripts/SFEARNet/LEVIR_CD_256/train_default.sh`
     - `experiment_scripts/SFEARNet/WHU-CD/train_default.sh`
     - `experiment_scripts/SFEARNet/CLCD/train_default.sh`
     but they currently rely on default `alpha/beta/gamma` values and do not pass these coefficients explicitly.
- Important correction:
  - Earlier notes and manuscript wording describe a two-layer `1x1 conv + BN + ReLU` feature projection head.
  - Current active code in `loss/ContrastiveLoss.py` has those projection layers commented out.
  - Therefore the active implementation is:
    1) take `feat1` and `feat2` from SFEARNet;
    2) apply channel-wise L2 normalization directly;
    3) compute pixel-wise Euclidean distance;
    4) apply label-guided contrastive loss.
  - This issue was later resolved in Update 34:
    - keep the direct-feature implementation;
    - revise paper text to remove the projection-head claim.
- Additional reproducibility notes:
  - `margin=3.0` remains hard-coded in `train.py`.
  - Because L2-normalized Euclidean feature distance is theoretically bounded near `2`, this setting still requires justification or ablation.
  - `infer_IOU.py`, `infer_WHCD_IOU.py`, and `infer_CLCD_IOU.py` contain hard-coded dataset/model paths; they are useful for current local experiments but should be parameterized before formal reproducibility.
  - `.gitignore` currently ignores `manuscript/`, so manuscript edits are not visible in normal `git status`.
- Next recommended step:
  1) Follow Update 34's decision to keep direct feature normalization and distance constraint.
  2) Expose `margin` as a runtime argument before running sensitivity experiments.
  3) Update dataset scripts to explicitly record `alpha`, `beta`, `gamma`, and eventually `margin`.
  4) Continue manuscript work with priority on `Introduction` and `Experiment`.

### 2026-05-09 (Update 34)
- User confirmed the implementation/paper direction:
  - use direct L2 normalization and distance constraint on the bi-temporal deep features;
  - do not use the previously discussed `1x1 conv + BN + ReLU` projection head as part of the current method.
- Updated consistency documents and drafting sources:
  1) `manuscript/Sections/4-Methodology.tex`
     - removed the projection-head formula and algorithm step;
     - Algorithm 1 now directly normalizes `F^{t_1}` and `F^{t_2}` before distance computation.
  2) `recode.txt`
     - revised Algorithm 1 description, steps, and formulas to match direct feature normalization.
  3) `PROJECT_PROGRESS.md`
     - corrected the current method summary at the top of this file.
  4) `HANDOVER_PROMPT.txt`
     - will be kept aligned with the fixed decision.
- Current Algorithm 1 after this decision:
  1) input `F_A`, `F_B`, and label `Y`;
  2) compute `f_A = Norm(F_A)` and `f_B = Norm(F_B)`;
  3) compute `d_ij = ||f_A(i,j) - f_B(i,j)||_2`;
  4) apply label-guided contrastive loss:
     - unchanged pixels are pulled closer;
     - changed pixels are separated by margin `m`;
  5) average, normalize, and clamp the contrastive loss.
- Next recommended step:
  1) expose `margin` as a command-line argument;
  2) revise any remaining PPT/manuscript text that still mentions feature projection;
  3) continue `Introduction` and `Experiment` after this code-paper consistency issue is now fixed.

### 2026-05-09 (Update 35)
- User decided that the paper should present one integrated algorithm rather than two separate algorithms.
- Updated `manuscript/Sections/4-Methodology.tex`:
  1) changed the Methodology narrative from "two concrete algorithms" to one unified annotation-consistent optimization procedure;
  2) renamed the formula subsections so they explain mechanisms without being labeled Algorithm 1 / Algorithm 2;
  3) removed the standalone contrastive-supervision algorithm environment;
  4) kept one final algorithm environment:
     - `Annotation-Consistent Bounded Multi-Task Optimization`;
  5) the single algorithm now covers:
     - dual-temporal input and network outputs;
     - direct L2 normalization of bi-temporal deep features;
     - pixel-wise distance and label-guided contrastive loss;
     - segmentation, edge, and contrastive branch normalization;
     - `alpha/beta/gamma` weighted fusion;
     - final bounded loss and backpropagation.
- Updated `recode.txt`:
  - replaced the old two-algorithm record with a single paper-ready algorithm and formula record.
- Updated `HANDOVER_PROMPT.txt`:
  - added the one-algorithm organization rule for future sessions.
- Verification:
  - checked `manuscript/Sections/4-Methodology.tex`, `recode.txt`, and `HANDOVER_PROMPT.txt`;
  - no remaining old references to `alg:contrastive`, `alg:joint_optimization`, or two-algorithm wording in the active Methodology/recode files.
- Current writing decision:
  - Method section can still discuss two internal mechanisms, but the paper should show only one algorithm block to make the method appear more integrated and concise.
- Next recommended step:
  1) expose `margin` as a command-line parameter;
  2) continue writing `Introduction` and `Experiment`;
  3) later compile the manuscript in a TeX-enabled environment to verify algorithm float placement.

### 2026-05-09 (Update 36)
- User requested an overall description of the unified algorithm.
- Fixed explanation direction:
  - describe the method as one end-to-end annotation-consistent training algorithm;
  - start from dual-temporal input and network outputs;
  - then explain direct L2-normalized feature distance constraint;
  - then explain segmentation, edge, and contrastive branch normalization;
  - finally explain weighted bounded fusion and backpropagation.
- Recommended naming remains:
  - `Annotation-Consistent Bounded Multi-Task Optimization`
- Writing emphasis:
  - the algorithm is not a new backbone structure;
  - its core is using label-guided feature distance supervision plus bounded multi-task joint optimization to suppress pseudo-change responses.

### 2026-05-09 (Update 37)
- User asked whether the unified algorithm description has been written into the paper Methodology chapter.
- Checked `manuscript/Sections/4-Methodology.tex`.
- Confirmed current Methodology already contains:
  1) an overall statement that the method is a unified annotation-consistent optimization procedure;
  2) the label-guided pixel-wise feature contrastive supervision formulas;
  3) the bounded multi-task joint optimization formulas;
  4) one single algorithm environment:
     - `Annotation-Consistent Bounded Multi-Task Optimization`.
- Note:
  - the previous chat answer's Chinese explanatory paragraph was not inserted verbatim;
  - the paper chapter currently contains the formal English journal-style version.

### 2026-05-09 (Update 38)
- User provided a PDF screenshot of the compiled algorithm and requested journal-style polishing.
- Issues identified from screenshot:
  1) `network` lacked a mathematical symbol;
  2) outputs `P`, `P^e`, `F^{t_1}`, `F^{t_2}` needed explicit names;
  3) the pixel loop needed to state the traversed domain;
  4) algorithm steps for distance, contrastive penalty, aggregation, branch loss, normalization, and fusion needed names;
  5) branch loss and normalization steps needed equation references;
  6) wording like "Fuse segmentation terms" should be more professional.
- Completed edits in `manuscript/Sections/4-Methodology.tex`:
  1) added network notation `\mathcal{N}_{\theta}`;
  2) added network mapping equation:
     - `(\mathbf{P}, \mathbf{P}^{e}, \mathbf{F}^{t_1}, \mathbf{F}^{t_2}) = \mathcal{N}_{\theta}(\mathbf{I}^{t_1}, \mathbf{I}^{t_2})`;
  3) defined:
     - `\mathbf{P}` as change prediction logits;
     - `\mathbf{P}^{e}` as edge prediction logits;
     - `\mathbf{F}^{t_1}, \mathbf{F}^{t_2}` as bi-temporal deep features;
  4) defined spatial domain:
     - `\Omega={1,...,H}\times{1,...,W}`;
     - algorithm loop now uses `for each pixel location (i,j)\in\Omega`;
  5) added labels to all relevant formulas:
     - network mapping,
     - feature normalization,
     - feature distance,
     - pixel-wise contrastive loss,
     - image-level contrastive aggregation,
     - contrastive normalization,
     - branch loss computation,
     - CE/IoU/edge normalization,
     - segmentation fusion,
     - total objective,
     - coefficient constraints,
     - final objective;
  6) rewrote the algorithm steps into professional named operations:
     - Network inference;
     - Feature normalization;
     - Feature discrepancy estimation;
     - Contrastive penalty assignment;
     - Contrastive branch aggregation;
     - Branch loss computation;
     - Bounded branch normalization;
     - Segmentation loss fusion;
     - Multi-task objective fusion;
     - Bounded objective update.
- Formatting note:
  - algorithm lines were shortened to reduce awkward wrapping in the PDF.

### 2026-05-09 (Update 39)
- User asked to replace the first subformula in the branch-loss equation with the explicit formula of `torch.nn.CrossEntropyLoss()` from `train.py`.
- Updated `manuscript/Sections/4-Methodology.tex`:
  - replaced `\mathcal{L}_{ce}=\operatorname{CE}(\mathbf{P},\mathbf{Y})` with the expanded pixel-wise cross-entropy formula:
    - average over spatial domain `\Omega`;
    - softmax over binary classes `c\in\{0,1\}`;
    - true-class log probability indexed by `y_{ij}`.
- Kept the IoU and edge loss definitions in the same branch-loss equation.
- Used an `aligned` equation layout to avoid an overlong single-line formula in the compiled PDF.

### 2026-05-09 (Update 40)
- User asked for the exact mathematical formulas corresponding to `IoULoss()` and `Diceloss()` used in `train.py`.
- Confirmed code-level definitions:
  1) `IoULoss()` in `loss/Iouloss.py`:
     - applies `softmax` to logits;
     - converts labels to one-hot format;
     - computes soft IoU over foreground class(es) only;
     - returns `1 - IoU`.
  2) `Diceloss()` in `loss/Diceloss.py`:
     - applies `softmax` to logits;
     - converts labels to one-hot format;
     - computes soft Dice over foreground class(es) only;
     - returns `1 - Dice`.
- Important implementation detail:
  - `train.py` passes `num_classes=2`, so the effective computation only uses the foreground class `c=1`.
  - Both losses are computed over batch and spatial dimensions with a numerical stabilizer `1e-7`.

### 2026-05-09 (Update 41)
- User asked the assistant to quickly understand the previous work from:
  - `PROJECT_PROGRESS.md`
  - `HANDOVER_PROMPT.txt`
- Re-read both handover files and re-checked the key implementation files required by the handover prompt:
  - `loss/ContrastiveLoss.py`
  - `utils/stages.py`
  - `train.py`
  - `utils/args_utils.py`
  - `model/SFEARNet.py` output locations
  - active Methodology/recode references for projection-head wording
- Current understanding confirmed:
  1) The project is a remote-sensing change detection paper/code project centered on pseudo-change suppression and annotation-consistent learning.
  2) The baseline network structure is inherited from SFEARNet and should not be claimed as the main innovation.
  3) The current innovations are:
     - direct L2-normalized bi-temporal deep feature distance supervision with label-guided contrastive loss;
     - bounded three-branch multi-task joint optimization over segmentation, edge, and contrastive branches.
  4) The paper organization decision is now one integrated algorithm:
     - `Annotation-Consistent Bounded Multi-Task Optimization`.
  5) Current active code does not use the previously discussed projection head; Methodology and `recode.txt` have already been adjusted to direct feature normalization.
  6) `margin=3.0` is still hard-coded in `train.py`; `alpha`, `beta`, and `gamma` are exposed in `utils/args_utils.py`.
- Conclusion:
  - The method definition and Methodology chapter are mostly stable and implementation-consistent.
  - The next fragile points are reproducible experiments, margin exposure/ablation, and completing Introduction/Experiment writing.
- Next recommended step:
  1) expose `margin` as a command-line argument;
  2) update dataset training scripts to explicitly record `alpha`, `beta`, `gamma`, and `margin`;
  3) continue manuscript work on `Introduction` and `Experiment`;
  4) optionally verify the expanded IoU/Dice formulas in Methodology against `loss/Iouloss.py` and `loss/Diceloss.py` before final submission.

### 2026-05-09 (Update 42)
- User requested adding a subsection on common loss functions to Chapter 3 and moving the previous Formula 25 into that subsection.
- Completed manuscript edits:
  1) Added `\subsection{Common Loss Functions}` to `manuscript/Sections/3-Preliminary.tex`.
  2) Moved the branch-loss definitions into Chapter 3 under label `eq:branch_losses`.
  3) Expanded the formula to explicitly define:
     - pixel-wise cross-entropy loss;
     - foreground IoU loss;
     - foreground Dice edge loss.
  4) Updated `manuscript/Sections/4-Methodology.tex` so the bounded multi-task joint optimization section references Eq. `eq:branch_losses` instead of redefining the basic losses.
- Verification:
  - `eq:branch_losses` is now defined once in Chapter 3 and referenced in Chapter 4 and the algorithm.
  - The formulas were checked against `loss/Iouloss.py` and `loss/Diceloss.py` implementation logic: softmax probability, foreground class, and numerical stabilizer.
- Note:
  - Chapter 3 still contains remaining IEEE template formula examples after the newly added subsection; these can be removed later as part of manuscript cleanup.

### 2026-05-09 (Update 43)
- User asked to verify whether the three newly written common loss formulas are correct.
- Re-checked against implementation:
  1) `torch.nn.CrossEntropyLoss()` in `train.py`;
  2) `loss/Iouloss.py`;
  3) `loss/Diceloss.py`;
  4) calls in `utils/stages.py`, where IoU and Dice losses are both invoked with `num_classes=2`.
- Verification result:
  - The original formulas were conceptually correct but did not explicitly include the mini-batch dimension.
  - Updated `manuscript/Sections/3-Preliminary.tex` to a stricter implementation-consistent version:
    - defined foreground probabilities for change and edge logits using softmax;
    - wrote cross-entropy as averaged over `B|\Omega|`;
    - wrote IoU loss over the foreground class with batch and spatial summation;
    - wrote Dice edge loss over the foreground class with batch and spatial summation;
    - specified `\varepsilon=10^{-7}` and `\texttt{num\_classes=2}`.
- Current conclusion:
  - The three formulas now match the current code behavior for the binary setting used in training.

### 2026-05-09 (Update 44)
- User requested that the three common loss formulas in Chapter 3 should each have an independent equation number instead of sharing one equation number.
- Completed manuscript edits:
  1) Split the original shared `eq:branch_losses` aligned equation in `manuscript/Sections/3-Preliminary.tex` into three separate equation environments:
     - `eq:ce_loss` for cross-entropy loss;
     - `eq:iou_loss` for foreground IoU loss;
     - `eq:edge_loss` for foreground Dice edge loss.
  2) Kept `eq:foreground_probability` as the shared probability definition before the three loss equations.
  3) Updated `manuscript/Sections/4-Methodology.tex` references:
     - Methodology text now cites `Eqs.~\eqref{eq:ce_loss}--\eqref{eq:edge_loss}`;
     - Algorithm branch-loss computation line now cites the same equation range.
- Verification:
  - No remaining active references to old `eq:branch_losses`.
  - The three loss formulas now each receive their own equation number for easier lookup.

### 2026-05-09 (Update 45)
- User asked what Chapter 3 formulas (2)--(5) are used for.
- Confirmed current Chapter 3 equation order:
  1) Formula (1): pseudo-change definition;
  2) Formula (2): foreground probability definitions for change and edge logits;
  3) Formula (3): pixel-wise cross-entropy loss;
  4) Formula (4): foreground IoU loss;
  5) Formula (5): foreground Dice edge loss.
- Explanation fixed:
  - Formula (2) is an auxiliary softmax probability definition, not a loss function.
  - Formulas (3)--(5) are standard preliminary loss definitions used later by the bounded multi-task joint optimization in Chapter 4.

### 2026-05-09 (Update 46)
- User pointed out that the three small formulas in the old Methodology Formula (29) duplicate Chapter 3 formulas (3)--(5).
- Confirmed:
  - This judgment is correct for the old version where Methodology redefined CE, IoU, and Dice/edge losses.
  - The current LaTeX source has already removed the duplicated three-formula definition from `manuscript/Sections/4-Methodology.tex`.
  - Methodology now only references the preliminary definitions through:
    - `Eqs.~\eqref{eq:ce_loss}--\eqref{eq:edge_loss}`.
- Current writing rule:
  - Chapter 3 defines common/basic losses once.
  - Chapter 4 should focus on the proposed normalization, bounded fusion, coefficient constraints, and final multi-task joint optimization objective.
  - If the compiled PDF still shows the old Formula (29), it is from an old compilation and should be regenerated.

### 2026-05-10 (Update 47)
- User asked for a TGRS-suitable mathematical expression of cross-entropy loss.
- Recommended formula direction:
  - Use a mini-batch and spatial-domain averaged formulation consistent with `torch.nn.CrossEntropyLoss()`.
  - Define logits over binary classes `\mathcal{C}=\{0,1\}` and labels `y^b_{ij}`.
  - Write CE directly as the negative log-softmax probability of the ground-truth class.
- Suggested paper-ready formula:
  - `\mathcal{L}_{ce} = -\frac{1}{B|\Omega|}\sum_{b=1}^{B}\sum_{(i,j)\in\Omega}\log \frac{\exp(P_{b,y^b_{ij},i,j})}{\sum_{c\in\mathcal{C}}\exp(P_{b,c,i,j})}`.
- Note:
  - This is appropriate for the current binary change detection setting and matches the implementation because PyTorch `CrossEntropyLoss` takes logits and internally applies log-softmax.

### 2026-05-10 (Update 48)
- User asked what Chapter 3 formula (2) means.
- Clarified interpretation:
  - Formula (2) is not a loss function.
  - It defines the foreground probabilities after applying softmax to the change-prediction logits and edge-prediction logits.
  - `p^1_{b,ij}` is the foreground probability of the change branch.
  - `q^1_{b,ij}` is the foreground probability of the edge branch.
  - These probabilities are the inputs required by the IoU and Dice formulas that follow.
- Writing rule fixed:
  - Formula (2) should be described as a probability-mapping definition, not as a supervisory loss.

### 2026-05-10 (Update 49)
- User asked whether `i, j` in the formula represent pixels.
- Clarified notation:
  - Yes, `(i,j)` denotes a pixel location in the spatial domain `\Omega`.
  - `i` and `j` are the row and column indices of a pixel in the image plane.
  - In the current formulas, `b` is the batch index, while `(i,j)` indexes spatial positions.
- Writing rule fixed:
  - When explaining the formulas in the paper, explicitly state that `\Omega=\{1,\ldots,H\}\times\{1,\ldots,W\}` and each `(i,j)\in\Omega` corresponds to one pixel location.

### 2026-05-10 (Update 50)
- User asked to rewrite the Chapter 3 loss-function explanation in a more TGRS-suitable style.
- Updated `manuscript/Sections/3-Preliminary.tex`:
  - rewrote the opening paragraph of `Common Loss Functions` in more formal journal language;
  - explicitly stated that `\Omega` is the spatial domain of an image and `(i,j)` denotes a pixel location;
  - described `\mathbf{P}` and `\mathbf{P}^{e}` as the logits of the change and edge branches;
  - described `\mathbf{Y}` and `\mathbf{Y}^{e}` as the corresponding binary supervision maps;
  - tightened the wording of the CE, IoU, and Dice loss descriptions.
- Result:
  - the section is now more consistent with TGRS-style mathematical exposition while keeping the formulas unchanged.

### 2026-05-10 (Update 51)
- User requested a notation table in the first subsection of Chapter 3, preceded by a short opening paragraph.
- Completed edits in `manuscript/Sections/3-Preliminary.tex`:
  1) Added a concise introductory sentence before the table to explain that the main symbols are summarized for consistency.
  2) Inserted `Table~\ref{tab:notations}` under `\subsection{Notations}`.
  3) The table records the meaning of the main symbols used across the manuscript, including:
     - bi-temporal inputs;
     - predicted and ground-truth maps;
     - logits and feature maps;
     - spatial domain and pixel index;
     - foreground probabilities;
     - batch size;
     - loss symbols and optimization coefficients.
- Current status:
  - The notation table is now available for reuse in later manuscript sections to keep symbol definitions consistent.

### 2026-05-10 (Update 52)
- User provided a compiled PDF screenshot showing that Chapter 3 loss formulas overflowed the IEEE/TGRS two-column layout and overlapped with right-column text.
- Cause identified:
  - The IoU and Dice formulas used long single-line fractions whose denominators exceeded the column width.
  - Formula (2) also placed two softmax fractions on the same line, which is fragile in two-column layout.
- Completed edits in `manuscript/Sections/3-Preliminary.tex`:
  1) Rewrote Formula (2) as a two-line aligned equation for `p^1_{b,ij}` and `q^1_{b,ij}`.
  2) Introduced compact summation notation:
     - `\sum_{b,ij}` denotes `\sum_{b=1}^{B}\sum_{(i,j)\in\Omega}`.
  3) Rewrote IoU loss with local auxiliary terms:
     - `I_{\text{chg}}` for foreground intersection;
     - `U_{\text{chg}}` for foreground union.
  4) Rewrote Edge/Dice loss with local auxiliary terms:
     - `I_{\text{edge}}` for edge intersection;
     - `S_{\text{edge}}` for Dice denominator.
  5) Kept each loss as one independently numbered equation:
     - `eq:ce_loss`;
     - `eq:iou_loss`;
     - `eq:edge_loss`.
- Verification:
  - Source-level LaTeX structure and labels were checked.
  - Full PDF compilation could not be run in the current environment because neither `latexmk` nor `pdflatex` is installed.
- Additional note:
  - The screenshot also shows remaining IEEE template text in Chapter 3; this should be cleaned later for final manuscript quality.

### 2026-05-10 (Update 53)
- User asked how to format section/chapter references in the manuscript.
- Writing rule fixed for IEEE/TGRS style:
  - Use `Section~\ref{...}` for sections and subsections in the paper body.
  - Put `\label{...}` immediately after the corresponding `\section{...}` or `\subsection{...}` command.
  - Use non-breaking spaces `~` before references, e.g., `Section~\ref{sec:Preliminary}`.
  - Do not hard-code section numbers such as "Chapter 3" or "Section III" in the source.
  - Use `Eq.~\eqref{...}` for equations, `Table~\ref{...}` for tables, `Fig.~\ref{...}` for figures, and `Algorithm~\ref{...}` for algorithms.
- Current note:
  - Because the manuscript uses IEEEtran, top-level paper divisions should be called "sections" rather than "chapters" in English paper text.

### 2026-05-10 (Update 54)
- User asked how to handle BibTeX copied from Google Scholar when adding references to the manuscript.
- Confirmed repository layout:
  - `manuscript/references.bib` is the BibTeX database used by `\bibliography{references}`;
  - `manuscript/Sections/9-Biography.tex` is only for author biographies and should not contain bibliography entries.
- Practical writing rule:
  - Google Scholar BibTeX entries can usually be pasted into `manuscript/references.bib` with minimal editing.
  - Check entry type, citation key, author formatting, title capitalization, venue fields, year, pages, DOI, and remove unnecessary metadata if needed.

### 2026-05-10 (Update 55)
- User provided a Google Scholar BibTeX entry for `PASSNet` and asked whether it can be pasted directly.
- Recommended handling:
  - Paste the entry into `manuscript/references.bib`, not `Sections/9-Biography.tex`.
  - Keep the citation key unique, e.g. `ji2023passnet`.
  - Protect acronym capitalization in the title if needed, e.g. by bracing `PASSNet`.
  - `publisher={IEEE}` is optional for a journal article and can be removed for cleanliness.
  - If available, add DOI or early-access metadata from the publisher page.
- Citation rule:
  - Cite in text using `\cite{ji2023passnet}` after the entry is added to the BibTeX file.

### 2026-05-10 (Update 56)
- User requested a full Related Work section written in LaTeX, strictly following TGRS style and based on `manuscript/references.bib`.
- Completed edits in `manuscript/Sections/2-Related_Work.tex`:
  1) Rewrote the placeholder file into a complete section.
  2) Added four subsections exactly as requested:
     - visual representation learning in remote sensing;
     - remote sensing change detection;
     - pseudo-change in remote sensing change detection;
     - semantic segmentation in remote sensing.
  3) Kept the text in formal journal style, with citation-based discussion and no bullet formatting.
  4) Renamed the section label to `sec:RelatedWork` and added subsection labels for later reuse.
- Verification:
  - All citations used in the new section are drawn from `manuscript/references.bib`.
  - The old placeholder text was removed.
- Note:
  - Full LaTeX compilation was not run in this environment, so final typesetting should still be checked in the user's TeX setup.

### 2026-05-10 (Update 57)
- User requested that the Related Work section be written strictly based on the existing references and in a TGRS-style tone.
- Further refined `manuscript/Sections/2-Related_Work.tex`:
  - kept the four-subsection structure;
  - tightened the opening and transition sentences;
  - added more directly relevant citations from `manuscript/references.bib`, including attention-based change detection and feature interaction references;
  - improved the pseudo-change subsection to better connect edge-aware modeling, content cleansing, and semi-supervised contrastive strategies.
- Current status:
  - the related work section now reads as a journal-style literature review rather than a placeholder outline.

### 2026-05-10 (Update 58)
- User requested further supplementation of each subsection in the Related Work section based on the existing references.
- Rewrote `manuscript/Sections/2-Related_Work.tex` into a fuller TGRS-style literature review with four subsections:
  1) visual representation learning in remote sensing;
  2) remote sensing change detection;
  3) pseudo-change in remote sensing change detection;
  4) semantic segmentation in remote sensing.
- Main content additions:
  - strengthened the discussion of spatial-spectral modeling and general remote sensing representation learning;
  - expanded the change detection subsection to cover Siamese networks, attention, multi-scale fusion, transformers, edge-aware refinement, and feature interaction;
  - expanded the pseudo-change subsection to cover edge-aware modeling, content cleansing, building-centric semantic comparison, and semi-supervised/contrastive strategies;
  - refined the segmentation subsection to emphasize dense prediction, foreground saliency, and its relevance to boundary-aware change analysis.
- Current status:
  - the Related Work section now contains substantive literature review text rather than only a structural outline.

### 2026-05-10 (Update 59)
- User asked to further polish the Related Work section based on the current target length.
- Rewrote `manuscript/Sections/2-Related_Work.tex` into a more complete TGRS-style literature review while keeping the four requested subsections unchanged.
- Current section structure and emphasis:
  1) visual representation learning in remote sensing:
     - expanded discussion of hyperspectral classification, object detection, anomaly detection, super-resolution, and spatial-spectral cooperation;
  2) remote sensing change detection:
     - expanded discussion of Siamese networks, attention, self-adaptation, deep supervision, multi-scale fusion, transformers, edge-aware refinement, and SFEARNet;
  3) pseudo-change:
     - expanded discussion of edge-aware mitigation, content cleansing, building-centric semantics, semi-supervised learning, and contrastive learning;
  4) semantic segmentation:
     - kept the section concise but formal, emphasizing foreground saliency, boundary modeling, and its relevance to change detection.
- Current status:
  - the Related Work section now has a more journal-like density and is no longer just a structural draft.

### 2026-05-11 (Update 60)
- User requested another revision of the Related Work section with three constraints:
  1) shorten the length: one paragraph per subsection, except the pseudo-change subsection with two paragraphs;
  2) use a point-to-point literature review style, e.g., explicitly state what each representative paper contributes;
  3) reassess subsection titles because the first and fourth subsections seemed overlapping.
- Completed edits in `manuscript/Sections/2-Related_Work.tex`:
  - rewrote the section into a more concise four-subsection structure;
  - changed subsection titles to reduce overlap:
    1) `Visual Representation Learning for Remote Sensing`;
    2) `Bi-Temporal Remote Sensing Change Detection`;
    3) `Pseudo-Change Suppression in Change Detection`;
    4) `Dense Prediction and Semantic Segmentation in Remote Sensing`;
  - made the writing more point-to-point by associating representative papers with their specific contributions;
  - kept all citations drawn from `manuscript/references.bib`.
- Current structure:
  - subsection 1: one paragraph;
  - subsection 2: one paragraph;
  - subsection 3: two paragraphs;
  - subsection 4: one paragraph.
- Rationale:
  - subsection 1 now covers general remote sensing visual representation across classification/detection/restoration;
  - subsection 4 now covers dense prediction, foreground modeling, and semantic segmentation as design references for pixel-level change detection, so it no longer duplicates subsection 1.

### 2026-05-11 (Update 61)
- User clarified that point-to-point writing should group papers with similar contributions together, rather than mechanically introducing every paper one by one.
- Updated `manuscript/Sections/2-Related_Work.tex` accordingly:
  - grouped related works by research point within each subsection;
  - examples:
    - weakly supervised detection and full-scale detection are grouped as object-level interpretation;
    - SNUNet-CD and SASiamNet are grouped as Siamese/dual-stream improvements;
    - attention-difference and deep-supervision methods are grouped as discrimination/supervision enhancement;
    - transformer and edge-assisted methods are grouped as contextual/boundary modeling;
    - semi-supervised and teacher-model methods are grouped as limited-supervision robustness;
    - multi-scale dense prediction methods are grouped as dense prediction and boundary-aware design.
- Current status:
  - The Related Work section now uses a "research point -> representative works -> shared contribution/difference -> remaining gap" structure.

### 2026-05-11 (Update 62)
- User provided a PDF screenshot showing text overlap near the top of the first page.
- Diagnosis:
  - The overlap was caused by IEEE copyright/pubid markup.
  - `\IEEEpubid{...}` in `manuscript/Manuscript.tex` and `\IEEEpubidadjcol` in `manuscript/Sections/3-Preliminary.tex` were active at the same time, creating a layout conflict in the current draft.
- Completed fix:
  1) Commented out `\IEEEpubid{...}` in `manuscript/Manuscript.tex`.
  2) Removed `\IEEEpubidadjcol` from `manuscript/Sections/3-Preliminary.tex`.
- Current status:
  - The first-page overlap should disappear after recompilation.
  - If IEEE copyright text is needed later for submission, it should be re-enabled with the correct pubid placement and column adjustment on the first page only.

### 2026-05-11 (Update 63)
- User asked whether the recent fix to the first-page overlap was the correct change.
- Clarified decision:
  - For the current draft, disabling `\IEEEpubid{...}` and removing `\IEEEpubidadjcol` is the correct way to eliminate the overlap.
  - This is a layout fix, not a change to the manuscript content.
  - If the final submission later requires IEEE copyright/pubid markup, the command should be restored and placed correctly on the first page.
- Current rule:
  - Draft version: keep pubid markup disabled to avoid overlap.
  - Final submission version: re-enable pubid only if required by the target IEEE workflow, and then re-check the first-page layout.

### 2026-05-11 (Update 64)
- User requested reverting the first-page copyright/pubid fix back to the previous version.
- Reverted the temporary layout fix:
  1) Restored `\IEEEpubid{...}` in `manuscript/Manuscript.tex`.
  2) Kept the original pubid-related comment guidance.
  3) Restored the previous `manuscript/Sections/3-Preliminary.tex` state before the temporary removal of `\IEEEpubidadjcol`.
- Current status:
  - The manuscript is back to the earlier layout state requested by the user.

### 2026-05-14 (Update 65)
- Completed handover-context alignment by reading:
  - `PROJECT_PROGRESS.md`
  - `HANDOVER_PROMPT.txt`
- Consolidated current project understanding:
  1) Baseline part is the SFEARNet backbone/pipeline (semantic flow + edge-aware prediction + encoder-decoder flow), which is not claimed as new contribution.
  2) Current new contributions are limited to:
     - label-guided pixel-level feature contrastive constraint in `loss/ContrastiveLoss.py`;
     - bounded three-branch multi-task joint optimization in `utils/stages.py`.
  3) Method objective remains annotation-consistent change detection under pseudo-change disturbance, not pure visual-difference response.
- Confirmed writing/consistency rules from handover:
  - remove projection-head wording from paper/PPT/algorithm description;
  - use "multi-task joint optimization" terminology;
  - keep single algorithm narrative: `Annotation-Consistent Bounded Multi-Task Optimization`.
- Next-step priority remains:
  1) clean any residual projection-head statements in manuscript materials;
  2) expose `margin` as CLI argument for reproducible ablation;
  3) organize `alpha/beta/gamma/margin` ablation scripts and tables;
  4) continue Method/Experiment writing centered on pseudo-change and annotation consistency.

### 2026-05-14 (Update 66)
- User requested continued maintenance of both handover files.
- Confirmed collaboration rule:
  - keep synchronizing `PROJECT_PROGRESS.md` and `HANDOVER_PROMPT.txt` in subsequent rounds;
  - each round should leave concise records of what was done, key conclusions, and next actionable step.
- Current status:
  - handover documentation remains active and ready for continuous session transfer.

### 2026-05-14 (Update 67)
- User requested polishing `Algorithm 1` in Methodology with five constraints:
  1) align better with TGRS-style algorithm writing;
  2) reduce dense equation referencing;
  3) add variable annotations inside the algorithm;
  4) avoid repetitive/mechanical sentence patterns;
  5) make the flow more natural and readable.
- Completed edit:
  - Rewrote algorithm block in `manuscript/Sections/4-Methodology.tex` (`Algorithm~\ref{alg:proposed}`).
  - Added explicit input/output and in-algorithm variable notes (`\mathbf{P}, \mathbf{P}^e, \mathbf{F}^{t_1}, \mathbf{F}^{t_2}, \Omega`).
  - Replaced formula-reference-heavy steps with action-oriented procedural descriptions.
  - Kept only key equation links (network mapping, contrastive aggregation, final fusion), and removed unnecessary repeated equation citations.
  - Reorganized contrastive loop using semantic actions: unchanged pixels are pulled together; changed pixels are pushed to margin.
- Verification status:
  - LaTeX compile was not executed due to missing local TeX tools (`latexmk` and `pdflatex` not found in environment).
- Next step:
  1) compile on a TeX-enabled machine and check line breaks/overflow inside Algorithm 1;
  2) if needed, shorten variable-note line or split into two `\STATE` lines to fit IEEE column width.

### 2026-05-15 (Update 68)
- User asked for the meaning of `y_{ij}=0` in Algorithm 1.
- Clarified notation:
  - `y_{ij}` is the ground-truth pixel label at location `(i,j)` in the change map `\mathbf{Y}`;
  - `y_{ij}=0` means the pixel is labeled as unchanged;
  - `y_{ij}=1` means the pixel is labeled as changed.
- Current algorithm rule:
  - unchanged pixels use the attraction term `\frac{1}{2}d_{ij}^2`;
  - changed pixels use the margin term `\frac{1}{2}\max(0,m-d_{ij})^2`.

### 2026-05-17 (Update 69)
- User asked the assistant to read `PROJECT_PROGRESS.md` and `HANDOVER_PROMPT.txt` to understand the current project progress.
- Completed handover review and re-checked the current key implementation files:
  - `model/SFEARNet.py`
  - `loss/ContrastiveLoss.py`
  - `utils/stages.py`
  - `train.py`
  - `utils/args_utils.py`
- Current code-consistency conclusion:
  1) SFEARNet still returns four outputs during training: change logits, edge logits, first-temporal deep feature, and second-temporal deep feature.
  2) `ContrastiveLoss` still does not execute the old two-layer `1x1 conv + BN + ReLU` projection head; those layers remain commented out.
  3) The active contrastive branch directly L2-normalizes `feat1` and `feat2`, computes pixel-wise Euclidean distance, pulls unchanged pixels together, and separates changed pixels with margin.
  4) `utils/stages.py` still uses bounded three-branch multi-task joint optimization: CE/IoU segmentation branch, contrastive branch, and edge branch.
  5) `train.py` still fixes `margin=3.0`; `utils/args_utils.py` exposes `alpha`, `beta`, and `gamma`, but not `margin`.
- Manuscript keyword check:
  - Current active Methodology/recode references align with direct feature normalization and margin-based contrastive supervision.
  - No new method-boundary change was found, so `HANDOVER_PROMPT.txt` was not modified in this round.
- Next actionable priorities:
  1) expose `margin` as a command-line argument before formal sensitivity experiments;
  2) prepare `alpha/beta/gamma/margin` ablation scripts and result tables;
  3) continue Experiment-section writing around pseudo-change suppression and annotation-consistent prediction;
  4) keep checking manuscript/PPT text for any residual projection-head wording before final submission.

### 2026-05-17 (Update 70)
- User requested a structural rewrite of Methodology section due to subsection overlap and unclear role boundaries.
- Completed major restructuring in `manuscript/Sections/4-Methodology.tex`:
  1) kept `Motivation` and `Overview`;
  2) removed overlapping old subsections (`Limitations...`, `Annotation-Consistency-Oriented Design`, old mixed loss/algorithm layout);
  3) added `Pseudo-Change Suppression via Label-Guided Contrastive Supervision` as a dedicated subsection focused on feature normalization, distance, margin-based contrastive penalty, and bounded contrastive loss;
  4) added `Objective Function` subsection to explicitly define branch fusion and final optimization target using `argmin`;
  5) moved algorithm `\label{alg:proposed}` into the last standalone subsection `Algorithm: Annotation-Consistent Bounded Multi-Task Optimization`, with a concise explanatory paragraph and cleaned action-style steps.
- Completed corresponding Preliminary cleanup in `manuscript/Sections/3-Preliminary.tex`:
  1) renamed `Common Loss Functions` to `Segmentation Loss Computation`;
  2) retained CE/IoU/Edge formal definitions and labels (`eq:ce_loss`, `eq:iou_loss`, `eq:edge_loss`);
  3) removed unrelated template-style math typography tutorial block that caused section-topic drift.
- Consistency conclusion:
  - Current Methodology now has clearer topic separation and reduced subsection overlap.
  - Segmentation-loss formulas are now located in Preliminary as requested, while Methodology references them and focuses on pseudo-change suppression + bounded joint objective + final algorithm.
  - Symbols and narrative remain aligned with `tab:notations` and current implementation semantics.
- Next step:
  1) compile and check cross-references/line breaks;
  2) if needed, tighten wording to match final journal style and remove remaining placeholder citations (`ref1`-`ref5`) in other sections.

### 2026-05-17 (Update 71)
- User requested restoring Algorithm 1 in Methodology from simplified flow to detailed version.
- Completed update in `manuscript/Sections/4-Methodology.tex`:
  1) kept the refactored subsection structure from Update 70;
  2) expanded algorithm subsection back to detailed form;
  3) reintroduced explicit variable annotations inside the algorithm;
  4) restored explicit contrastive subroutine with per-pixel normalization, distance, margin-based penalty, aggregation, and bounded normalization steps;
  5) aligned algorithm steps with key equations (`network_mapping`, `feature_normalization`, `feature_distance`, `pixel_contrastive`, `image_contrastive`, `contrastive_normalization`, `segmentation_fusion`, `total_objective`, `final_objective`).
- Conclusion:
  - Methodology keeps clear subsection boundaries while recovering algorithmic detail density required by user.
- Next step:
  1) compile and inspect line wrapping/overflow in Algorithm block;
  2) tune wording length if any step exceeds IEEE column width.

### 2026-05-18 (Update 72)
- User requested consistency check and alignment for `experiment_scripts` training entry scripts.
- Checked script coverage under `experiment_scripts/SFEARNet/`:
  - `CLCD/train_default.sh`
  - `WHU-CD/train_default.sh`
  - `LEVIR_CD_256/train_default.sh`
  - `WHU-CD/infer_default.sh` and `WHU-CD/infer_simple_IOU.sh` (both currently call `train.py`)
  - `GZ-CD/` directory exists but currently contains no script files.
- Implemented script alignment:
  - Updated `experiment_scripts/SFEARNet/LEVIR_CD_256/train_default.sh` to include DataLoader controls:
    - added variables: `NUM_WORKERS=4`, `PIN_MEMORY=true`, `PERSISTENT_WORKERS=true`;
    - added CLI args: `--num_workers`, `--pin_memory`, `--persistent_workers`.
- Consistency conclusion (train scripts):
  - DataLoader parameter passing is now aligned across all three `train_default.sh` files (`CLCD`, `WHU-CD`, `LEVIR_CD_256`).
- Residual inconsistency note:
  - `infer_default.sh` and `infer_simple_IOU.sh` are still named as infer scripts but execute `train.py`; this remains a separate cleanup task.
- Next suggested step:
  1) split infer scripts to true inference entrypoints and separate from training templates;
  2) optionally make `alpha/beta/gamma` explicit in all train scripts for stronger reproducibility.

### 2026-05-18 (Update 73)
- User explicitly requested that every assistant run maintain two project-state files:
  - `PROJECT_PROGRESS.md`
  - `HANDOVER_PROMPT.txt`
- Confirmed current repository state:
  - both files already exist in the project root;
  - both files are currently untracked by Git;
  - `HANDOVER_PROMPT.txt` already contains a persistent-recording rule under the handover notes.
- Maintenance rule for future rounds:
  - update `PROJECT_PROGRESS.md` at the end of each run with what changed, current conclusions, and next steps;
  - update `HANDOVER_PROMPT.txt` when collaboration rules, method boundaries, implementation facts, or next-handoff instructions change;
  - if no handover-relevant fact changes, explicitly mention in `PROJECT_PROGRESS.md` that `HANDOVER_PROMPT.txt` did not require a substantive update.
- This round:
  - no code or manuscript logic was changed;
  - only the project-maintenance convention was reinforced.
- Next step:
  1) continue applying this file-maintenance convention after each subsequent coding, manuscript, or experiment-script update.

### 2026-05-18 (Update 74)
- User asked where the contrastive loss should be placed in Chapter-3 Fig.1 based on the current innovation code.
- Code-grounded verification:
  - model forward returns `x, edge, feat1, feat2` where `feat1/feat2` come from deepest backbone stage (`x1[3]`, `x2[3]`) and are upsampled to label resolution (`model/SFEARNet.py`).
  - contrastive branch consumes `feat1, feat2, label`, performs channel-wise L2 normalization and pixel-wise Euclidean distance, then margin-based penalty (`loss/ContrastiveLoss.py`).
  - final optimization fuses three normalized branches: segmentation, edge, contrastive (`utils/stages.py`).
- Figure placement conclusion:
  - contrastive loss should be drawn as a parallel supervision branch tapped from the deepest bi-temporal backbone features before decoder/head prediction, not from change-mask logits.
  - this branch should merge with CE/IoU and edge losses only at the final weighted objective node.
- Handover update note:
  - `HANDOVER_PROMPT.txt` does not need a substantive update this round because this placement is consistent with existing method-boundary records.
- Next step:
  1) if needed, refine Fig.1 labels to explicitly show `F_A^deep, F_B^deep -> MarginalDistance/Contrastive -> L_con^norm -> L_total`.

### 2026-05-19 (Update 75)
- User requested continued maintenance of both state files:
  - `PROJECT_PROGRESS.md`
  - `HANDOVER_PROMPT.txt`
- This round action:
  - performed routine handover maintenance and appended synchronized status records to both files;
  - kept method boundaries, optimization formulas, and writing constraints unchanged.
- Conclusion:
  - project handover state remains consistent; no new contradiction with existing method definition.
- Next step:
  1) continue same dual-file maintenance after each future code/manuscript/experiment change;
  2) only update `HANDOVER_PROMPT.txt` with substantive edits when rules/facts/boundaries change.

### 2026-05-19 (Update 76)
- User asked what the input to contrastive loss should be for methodology-figure drawing.
- Code-verified conclusion:
  - Contrastive loss input is `feat1`, `feat2`, and `label`:
    - `feat1`, `feat2`: deepest bi-temporal backbone features (`x1[3]`, `x2[3]`) upsampled to `(H, W)` in `model/SFEARNet.py`;
    - `label`: binary change map `Y` with shape `[B, H, W]`.
  - In training, it is called as `criterion_contrast(model_x[2], model_x[3], label)` in `utils/stages.py`.
  - Therefore, contrastive branch should be drawn from deep feature outputs, not from change-logit output.
- Handover note:
  - `HANDOVER_PROMPT.txt` receives a concise status append in this round for continuity.

### 2026-05-19 (Update 77)
- User requested baseline-model consolidation into the project `model/` directory, with one script per network class name for later unified training.
- Completed first-round migration (direct source copy into `model/`):
  - `model/USSFCNet.py`  <- `baselines/USSFC-Net/networks/USSFCNet.py`
  - `model/EATDer.py`  <- `baselines/EATDer/EATDernet.py`
  - `model/Net.py`  <- `baselines/DESSN/DESSN.py`
  - `model/CDNet.py`  <- `baselines/AMTNet/model/network.py`
  - `model/SiamUnet_diff.py`  <- `baselines/fully_convolutional_change_detection/siamunet_diff.py`
  - `model/SiamUnet_conc.py`  <- `baselines/fully_convolutional_change_detection/siamunet_conc.py`
- Name-conflict handling:
  - both BIT_CD and ChangeFormer define `BASE_Transformer`; to avoid filename collision in `model/`, they are currently staged as:
    - `model/BASE_Transformer_BIT.py`
    - `model/BASE_Transformer_ChangeFormer.py`
- Current status:
  - baseline model sources are now locally centralized under `model/` for next-step adapter/import cleanup.
- Next step:
  1) add minimal import-path fixes for copied files that still reference baseline-relative packages;
  2) create a unified model factory/registry in this repo to instantiate each baseline by name.

### 2026-05-19 (Update 78)
- User requested explicit decomposition for ambiguous baselines (STANet and MSCA-Net): one Python script per concrete network model under `model/`.
- Completed additional model-file extraction:
  - STANet-related network files:
    - `model/mynet3.py`
    - `model/CDSA.py` (from STANet `backbone.py`)
    - `model/BAM.py`
    - `model/PAM.py`
  - MSCA-Net registered-model files:
    - `model/Comprehensive_Atten_Unet.py`
    - `model/BCDU_net_D3.py`
    - `model/CPF_Net.py`
    - `model/CE_Net.py`
    - `model/msca_net.py` (contains `msca_net` and `msca_net_with_heatmap_output`)
- Current status:
  - ambiguous baselines are no longer treated as single black-box entries; concrete model scripts are now present in `model/`.
- Next step:
  1) normalize import dependencies for these copied files;
  2) build a unified baseline model registry/factory for your training framework.

### 2026-05-19 (Update 79)
- User requested packaging SFEARNet as an isolated subpackage under `model/` to separate it from newly added baseline model scripts.
- Completed restructuring:
  - created `model/sfearnet/` package;
  - moved/copied SFEARNet stack into package:
    - `SFEARNet.py`, `backbone.py`, `semantic_flow.py`, `edge_aware.py`, `pyramid.py`, `CBAM.py`;
  - added `model/sfearnet/__init__.py` exporting `SFEARNet`.
- Compatibility handling:
  - replaced top-level `model/SFEARNet.py` with a shim that re-exports `SFEARNet` from `model.sfearnet.SFEARNet`.
  - this keeps existing imports such as `from model.SFEARNet import SFEARNet` working in current train/infer scripts.
- Weight-file note:
  - copied existing `b0_backbone_weights.pth` into `model/sfearnet/model_data/`;
  - `b1_backbone_weights.pth` is not present in repository currently.
- Path fix:
  - updated `model/sfearnet/backbone.py` to resolve backbone weights via `os.path.dirname(__file__)` package-relative paths.
- Next step:
  1) decide whether to also update train/infer imports to the new canonical path `from model.sfearnet import SFEARNet`;
  2) optionally add `b1_backbone_weights.pth` if `phi='b1'` pretrained path is needed.

### 2026-05-19 (Update 80)
- User requested strict cleanup: keep SFEARNet definitions only inside `model/sfearnet/`, with main definition moved to `model/sfearnet/__init__.py`.
- Completed actions:
  1) moved SFEARNet implementation content into `model/sfearnet/__init__.py`.
  2) removed `model/sfearnet/SFEARNet.py`.
  3) updated training/inference imports from `from model.SFEARNet import SFEARNet` to `from model.sfearnet import SFEARNet` in:
     - `train.py`
     - `infer.py`
     - `infer_OA.py`
     - `infer_F1.py`
     - `infer_IOU.py`
     - `infer_WHCD_IOU.py`
     - `infer_CLCD_IOU.py`
  4) removed outer-layer SFEARNet-related scripts from `model/`:
     - `model/SFEARNet.py`
     - `model/backbone.py`
     - `model/semantic_flow.py`
     - `model/edge_aware.py`
     - `model/pyramid.py`
     - `model/CBAM.py`
     - `model/model_data/` (old location)
  5) kept only package-local SFEARNet stack under `model/sfearnet/`.
- Current status:
  - no SFEARNet definition remains outside `model/sfearnet/`.
- Note:
  - `model/sfearnet/model_data/` currently contains `b0_backbone_weights.pth` only.

### 2026-05-19 (Update 81)
- User requested final audit of `model/sfearnet/` to verify that SFEARNet can still be defined and imported as before.
- Completed checks:
  1) import-path consistency scan inside `model/sfearnet/`:
     - internal imports now use package-local relative imports (`from .xxx import ...`).
  2) structure check:
     - `model/sfearnet/` contains `__init__.py`, `backbone.py`, `semantic_flow.py`, `edge_aware.py`, `pyramid.py`, `CBAM.py`, and `model_data/`.
  3) syntax check:
     - `python3 -m py_compile` passed for all SFEARNet package scripts.
- Runtime validation blocker:
  - environment lacks `torch` (`ModuleNotFoundError: No module named 'torch'`), so live model instantiation/forward test could not be completed in this session.
- Current risk note:
  - `phi='b1'` pretrained path still requires `model/sfearnet/model_data/b1_backbone_weights.pth`, which is currently absent.

### 2026-05-19 (Update 82)
- User requested confirmation of SFEARNet I/O tensor shapes and reStructuredText-style function documentation in `model/sfearnet/`.
- Shape confirmation basis:
  - `data/Dataset.py`: image tensors are produced as `[B, 3, H, W]`; label/edge tensors are `[B, H, W]`.
  - `model/sfearnet/__init__.py::SFEARNet.forward`: returns `(x, edge, feat1, feat2)`.
- Added reST docstrings to `SFEARNet` class and `forward` method in `model/sfearnet/__init__.py`:
  - explicit input shape docs for `input1`, `input2`;
  - explicit output shape docs for `x`, `edge`, `feat1`, `feat2`;
  - note for deep-channel size dependency (`phi='b0' -> 256`, `phi='b1' -> 512`).
- Verification:
  - `python3 -m py_compile model/sfearnet/__init__.py` passed.

### 2026-05-19 (Update 83)
- User requested converting SFEARNet docstrings from reStructuredText to Google style for better VSCode recognition.
- Updated `model/sfearnet/__init__.py`:
  - converted `SFEARNet.__init__` docstring to Google style (`Args:`);
  - converted `SFEARNet.forward` docstring to Google style (`Args:`, `Returns:`, `Note:`), while preserving tensor-shape descriptions.
- Verification:
  - `python3 -m py_compile model/sfearnet/__init__.py` passed.

### 2026-05-19 (Update 84)
- User reported Google-style docstrings were still not clearly recognized in VSCode.
- Applied signature-level typing enhancement in `model/sfearnet/__init__.py` to improve Pylance/hover inference:
  - added `from typing import Tuple`;
  - added typed `__init__` signature;
  - added typed `forward` signature with explicit tuple return type:
    `Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]`.
- Verification:
  - `python3 -m py_compile model/sfearnet/__init__.py` passed.

### 2026-05-19 (Update 85)
- User requested implementing a unified baseline interface without modifying baseline model definitions.
- Added new adapter module:
  - `model/baseline_adapters.py`
- Implemented `UnifiedCDAdapter` with unified output contract:
  - forward returns `(logits, edge, feat1, feat2)`
  - for baselines lacking edge/features, returns `None` placeholders.
- Added default mapping rules for current baseline class names:
  - single-output models -> `(logits, None, None, None)`
  - `EATDer` pair output `(edge, block)` -> `(block, edge, None, None)`
  - `CDNet` triple output `(x, x8, x16)` -> `(x, None, None, None)`
  - `BASE_Transformer` variants -> tensor or list/tuple fallback to logits.
- Baseline source files were not modified.
- Verification:
  - `python3 -m py_compile model/baseline_adapters.py` passed.

### 2026-05-19 (Update 86)
- User requested adding function/class docstrings to `model/baseline_adapters.py` and applying this documentation standard by default in future edits.
- Completed:
  - added Google-style docstrings for:
    - `AdapterRule`
    - `_as_4d`
    - `_single_to_unified`
    - `_edge_logits_pair_to_unified`
    - `_triple_to_unified`
    - `_bit_changeformer_to_unified`
    - `UnifiedCDAdapter.__init__`
    - `UnifiedCDAdapter.forward`
    - `wrap_unified`
- Verification:
  - `python3 -m py_compile model/baseline_adapters.py` passed.

### 2026-05-19 (Update 87)
- User requested Chinese-language function docs while keeping variable names/types/syntax declarations in English.
- Completed documentation translation:
  1) `model/baseline_adapters.py`
     - translated all class/function docstrings to Chinese (Google style), while preserving English code symbols and type signatures.
  2) `model/sfearnet/__init__.py`
     - translated `SFEARNet` class docstring and `forward` docstring to Chinese;
     - preserved shape/type declarations and English identifiers.
- Verification:
  - `python3 -m py_compile model/baseline_adapters.py model/sfearnet/__init__.py` passed.

### 2026-05-19 (Update 88)
- User asked for a module name for the contrastive innovation block in methodology figure.
- Naming recommendation:
  - Preferred: `Label-Guided Feature Contrast Module (LGFCM)`.
- Alternative candidates:
  1) `Annotation-Consistent Contrastive Module (ACCM)`
  2) `Bi-Temporal Feature Contrast Module (BFCM)`
  3) `Pseudo-Change Suppression Contrast Module (PSCM)`
- Selection rationale:
  - explicitly reflects current code fact: supervision uses change labels and acts on deep bi-temporal features.

### 2026-05-19 (Update 88)
- User asked whether current `train.py` can still run SFEARNet normally after baseline-integration restructuring.
- Compatibility check results:
  - `train.py` now imports `SFEARNet` from `model.sfearnet` and still instantiates as before:
    - `model = SFEARNet(2, phi='b0', pretrained=True)`
  - all inference entries also import from `model.sfearnet` consistently.
  - no residual old-path import (`model.SFEARNet`, `model.backbone`, etc.) was found in active train/infer entry scripts.
  - required pretrained file for current train setting exists:
    - `model/sfearnet/model_data/b0_backbone_weights.pth`
- Conclusion:
  - For current setup (`phi='b0', pretrained=True`), SFEARNet call chain remains compatible with existing experiment flow.
- Residual risk:
  - `phi='b1'` with `pretrained=True` still requires missing `b1_backbone_weights.pth`.

### 2026-05-19 (Update 89)
- User asked whether AMTNet's main network should be considered `CDNet`, given multiple model files under `baselines/AMTNet/model/`.
- Code-grounded conclusion:
  - the active training/testing entrypoints (`baselines/AMTNet/train.py`, `baselines/AMTNet/test.py`) import and instantiate `CDNet` from `model/network.py`;
  - therefore, AMTNet's current main model for this project is `CDNet`.
- Clarification:
  - other files under `baselines/AMTNet/model/` (`siamunet_diff.py`, `siamunet_conc.py`, `unet.py`, `dtcdscn.py`, `MSCANet.py`) are additional candidate architectures or related implementations, but they are not the model currently used by the AMTNet training entry.

### 2026-05-20 (Update 90)
- User requested packaging CDNet as an isolated package under `model/`, including AMTNet-dependent modules.
- Completed refactor:
  - created `model/cdnet/` package;
  - placed CDNet main implementation into `model/cdnet/__init__.py`;
  - copied required dependencies from AMTNet source into package:
    - `model/cdnet/backbone.py`
    - `model/cdnet/modules.py`
- Compatibility handling:
  - `model/CDNet.py` is now a compatibility shim that re-exports `CDNet` from `model.cdnet`.
- Verification:
  - `python3 -m py_compile model/cdnet/__init__.py model/cdnet/backbone.py model/cdnet/modules.py model/CDNet.py` passed.

### 2026-05-20 (Update 91)
- User requested removing `model/CDNet.py` and making `model/baseline_adapters.py` rely on packaged `model/cdnet/` directly.
- Completed:
  1) deleted `model/CDNet.py`.
  2) updated `model/baseline_adapters.py`:
     - import path changed to `from model.cdnet import CDNet`;
     - CDNet default-rule dispatch changed to type-based check:
       `isinstance(base_model, CDNet)` -> `_cdnet_to_unified`.
  3) kept other baseline class-name-based rules unchanged.
- Verification:
  - `python3 -m py_compile model/baseline_adapters.py model/cdnet/__init__.py model/cdnet/backbone.py model/cdnet/modules.py` passed.

### 2026-05-21 (Update 75)
- User requested a detailed forward-flow analysis for drawing a more fine-grained SFEARNet diagram, specifically based on:
  - `model/sfearnet/__init__.py`
  - `model/sfearnet/backbone.py`
  - `model/sfearnet/pyramid.py`
- Confirmed implementation-aligned flow:
  - dual temporal inputs are independently encoded by shared `mit_b0/mit_b1` backbone into 4-scale features;
  - each same-scale pair is fused by `Pyramid_Merge` (concat-path + abs-diff-path + CBAM + Pyramid_Extraction);
  - fused multi-scale features are edge-guided (`Edge_Guidance_0/1`) into one edge branch plus refined pyramid features;
  - refined features enter SegFormer-style decode head for change logits, then are resampled to input resolution;
  - deepest original backbone features (`x1[3], x2[3]`) are separately upsampled and returned for contrastive supervision.
- Diagram implication:
  - contrastive branch should be drawn from the deepest backbone outputs (before pyramid/decode path), parallel to segmentation/edge prediction branches.
- Handover update note:
  - `HANDOVER_PROMPT.txt` is updated this round with forward-graph placement and branch dependency facts.
- Next step:
  1) if user wants, provide a box-by-box figure template with suggested node names and arrow labels for direct PPT/Visio drawing.

### 2026-05-21 (Update 76)
- User requested a more intuitive explanation of SFEARNet internals, especially:
  1) backbone structure,
  2) scale-wise bi-temporal fusion,
  3) edge-guided decoding path.
- Added implementation-grounded clarification after reading:
  - `model/sfearnet/backbone.py`
  - `model/sfearnet/pyramid.py`
  - `model/sfearnet/edge_aware.py`
  - `model/sfearnet/semantic_flow.py`
- Key clarified points for drawing:
  - backbone is a shared-weight dual-stream MiT encoder (4 stages) that outputs C1-C4 per timestamp;
  - `Pyramid_Merge` each scale contains two explicit paths (concat-attention path and abs-diff pyramid path) then residual-style summation;
  - edge guidance first predicts an edge map from shallow+deep features, then broadcasts this edge prior to all scales via pooling and multiplicative residual gating, followed by per-scale CBAM;
  - SegFormer decode uses semantic-flow-assisted multi-scale alignment to C1 before concat+fuse+prediction.
- Next step:
  1) provide a box-level flowchart template with suggested names and tensor-shape labels for direct plotting.

### 2026-05-21 (Update 77)
- User clarified requirement: needs structural flowchart (module internals), not tensor-size narration.
- Response focus adjusted to three diagram-ready blocks:
  1) generic backbone stage structure,
  2) scale-wise fusion (`Pyramid_Merge`) internal dual-path graph,
  3) edge-guidance + decoder structural graph.
- Output format switched to direct Mermaid flowcharts for immediate drawing.
- Handover note:
  - `HANDOVER_PROMPT.txt` updated to prioritize module-structure diagrams over shape tables when user requests architecture visualization.

### 2026-05-22 (Update 78)
- User asked whether the transformer attention in current SFEARNet backbone is a standard attention layer.
- Code-grounded conclusion:
  - It is not plain full global MHA used in vanilla ViT.
  - Query is computed on full tokens, while key/value optionally pass a spatial-reduction path (`sr_ratio > 1`) via strided conv before attention.
  - This matches SegFormer/MiT-style efficient attention (spatial-reduced attention), not the most naive standard self-attention.
- Next step:
  1) if needed, provide a side-by-side diagram: vanilla MHA vs MiT SR-Attention.

### 2026-05-22 (Update 79)
- User asked for a more explicit internal view of each Transformer Block, especially the exact normalization type.
- Clarified from `model/sfearnet/backbone.py`:
  - Block uses pre-norm twice (`norm1`, `norm2`), both are `nn.LayerNorm` from `norm_layer`.
  - In `mit_b0/mit_b1`, `norm_layer=partial(nn.LayerNorm, eps=1e-6)`.
  - Therefore normalization is LayerNorm (token-wise over channel dim), not BatchNorm.
- Response provides block-level flow including SR-Attention path and MLP(DWConv) path.

### 2026-05-22 (Update 80)
- User requested a more intuitive visualization of scale-wise bi-temporal fusion (`Pyramid_Merge`), similar to the Transformer Block presentation style.
- Added diagram-ready explanation with two levels:
  1) top-level dual-path merge graph;
  2) internal `Pyramid_Extraction` multi-branch graph.
- Clarified key semantics for plotting:
  - Path A models complementary bi-temporal context from concatenated features;
  - Path B models change saliency from absolute difference with multi-receptive-field extraction;
  - final output is residual-style additive fusion of both paths.

### 2026-05-22 (Update 81)
- User asked whether `AbsDiff Feature` is a backbone output.
- Clarification:
  - `AbsDiff Feature` is derived from backbone outputs, not a direct single-branch backbone output.
  - In `Pyramid_Merge`, it is computed as `input_abs = torch.abs(input1 - input2)` where `input1,input2` are same-scale features from two timestamps after backbone.
- Next step:
  1) keep distinction explicit in diagrams: backbone outputs vs derived difference features.

### 2026-05-22 (Update 82)
- User asked for a more intuitive explanation of how pyramid fusion consumes backbone outputs `C1~C4`.
- Clarified execution order in forward:
  - backbone outputs two 4-level lists: `[C1_t1,C2_t1,C3_t1,C4_t1]` and `[C1_t2,C2_t2,C3_t2,C4_t2]`;
  - pyramid fusion is pairwise and level-wise: `F1=py1(C1_t1,C1_t2)`, ..., `F4=py4(C4_t1,C4_t2)`;
  - each `py*` keeps that level’s spatial scale and outputs one fused feature for the same level;
  - fused list `[F1,F2,F3,F4]` then enters Edge_Guidance and decoder.
- Next step:
  1) if needed, provide a single-page end-to-end block diagram text template for direct drawing.

### 2026-05-22 (Update 83)
- User asked for internals of a single `Pyramid_Merge` block.
- Provided block-level explanation focusing on two-branch design and exact merge sequence:
  - Path A: concat pair -> DW/PW 1x1 conv stack -> CBAM;
  - Path B: abs diff -> Pyramid_Extraction(5 branches) -> residual add with abs diff;
  - Final: PathA + PathB(out) additive fusion.
- Next step:
  1) if needed, provide a publication-ready pseudo-equation panel for the same block.

### 2026-05-22 (Update 84)
- User asked where the feature/logits for segmentation loss are produced.
- Clarified supervision path in current implementation:
  - segmentation losses use `label_pred = model_x[0]` in `utils/stages.py`;
  - `model_x[0]` is `x` returned by `SFEARNet.forward`;
  - `x` is generated by `decode_head(edge-guided fused pyramid features)` then upsampled by `Resampler` (`self.re`) to input resolution;
  - CE/IoU are computed on this final change-branch logits against `label`.
- Handover note:
  - no new method-boundary change; HANDOVER_PROMPT kept unchanged this round.

### 2026-05-22 (Update 85)
- User asked whether edge loss and segmentation loss use the same feature.
- Clarification:
  - They share upstream fused pyramid features, but supervision is applied on different branch outputs.
  - Edge loss uses `model_x[1]` (`edge`) produced by `Edge_Guidance` edge branch and resampled by `re2`.
  - Segmentation loss uses `model_x[0]` (`x`) produced by `decode_head` on edge-guided features and resampled by `re`.
  - So: shared backbone/fusion context, different heads and different logits for loss computation.
- Handover note:
  - no rule/method-boundary change; HANDOVER_PROMPT unchanged this round.

### 2026-05-22 (Update 86)
- User asked whether change-segmentation loss is computed directly from 4 features.
- Clarification:
  - The 4 edge-guided features are decoder inputs, not final supervision tensors.
  - `decode_head` first fuses these 4 features and outputs a single change-logit map `x`.
  - Segmentation losses (CE/IoU) are computed on this fused `x` after resampling, not on each of the 4 features separately.
- Key code path:
  - `x = eg[1:] -> x = self.decode_head(x) -> x = self.re(x) -> return x`
  - `label_pred = model_x[0]; loss_ce/loss_iou(label_pred, label)`
- Handover note:
  - no new boundary/rule changes; HANDOVER_PROMPT not substantively updated.

### 2026-05-22 (Update 87)
- User asked what `decode_head` specifically is.
- Clarified:
  - `decode_head` is a SegFormer-style multi-scale decoder (`SegFormerHead_0` for b0, `SegFormerHead_1` for b1) defined in `model/sfearnet/__init__.py`.
  - It takes 4 edge-guided pyramid features, linearly embeds each level, aligns C2/C3/C4 to C1 scale with Semantic_flow + bilinear path, concatenates all levels, fuses by 1x1 conv, dropout, and final 1x1 classifier.
  - Output is one change-logit map later resampled and used by CE/IoU losses.
- Next step:
  1) if needed, provide a dedicated decode-head flowchart panel for direct insertion into chapter figures.

### 2026-05-24 (Update 88)
- User asked how many parts should be shown when drawing the decoder.
- Clarified decoder visualization recommendation:
  - use 4 main parts at the top level: multi-scale inputs, per-scale embedding, cross-scale alignment, fusion/classification.
  - optionally expand the alignment block to show Semantic_flow + bilinear path + residual addition for C2/C3/C4.
- Next step:
  1) provide a clean block diagram template with 4 main nodes and 1 optional expanded inset.

### 2026-05-24 (Update 92)
- User asked for the exact input/output of innovation code in `loss/ContrastiveLoss.py`.
- Confirmed from `ContrastiveLoss.forward(feat1, feat2, label)`:
  - inputs:
    - `feat1`: `[B, C, H, W]`
    - `feat2`: `[B, C, H, W]`
    - `label`: `[B, H, W]`, where `0=unchanged`, `1=changed`
  - output:
    - scalar loss tensor (0-dim), normalized and clamped to `[0,1]`.

### 2026-05-24 (Update 93)
- User asked for the meaning of `C/H/W` in tensor shapes.
- Clarified notation:
  - `C` = channel dimension
  - `H` = height
  - `W` = width
  - `B` = batch size
- Maintenance note:
  - this round also continued the required dual-file update workflow.

### 2026-05-24 (Update 94)
- User asked for the concrete size of `C/H/W` in the current project.
- Current project convention:
  - input image tensor: `3 x 256 x 256`
  - change/edge logits: `2 x 256 x 256`
  - contrastive deep feature channels: `256` for `phi='b0'`, `512` for `phi='b1'`
  - current experiments mostly use `H=W=256`

### 2026-05-24 (Update 95)
- User asked to trace SFEARNet step-by-step from image input to each intermediate output.
- Current verified flow (for `phi='b0'`, batch input, 256x256 images):
  1) Dataset loads bi-temporal images and labels:
     - `image_A`, `image_B`: `[B, 3, 256, 256]`
     - `label`, `edge`: `[B, 256, 256]`
  2) Backbone feature extraction:
     - `x1 = backbone(image_A)` and `x2 = backbone(image_B)`
     - each returns 4 feature maps:
       - stage1: `[B, 32, 64, 64]`
       - stage2: `[B, 64, 32, 32]`
       - stage3: `[B, 160, 16, 16]`
       - stage4: `[B, 256, 8, 8]`
  3) Pyramid merge on each level:
     - `x_0, x_1, x_2, x_3 = Pyramid_Merge(...)`
     - outputs keep the same spatial size as the corresponding stage, with channel compressed to the stage channel count.
  4) Edge guidance:
     - `eg = Edge_Guidance_0(x)`
     - returns edge prediction at coarse scale plus 4 refined feature maps.
  5) Decode head:
     - refined multi-scale features are fused by `SegFormerHead_0`
     - output logits are then resampled to `[B, 256, 256]` and passed through `re/re2` heads.
  6) Final forward output:
     - `x`: change logits `[B, 2, 256, 256]`
     - `edge`: edge logits `[B, 2, 256, 256]`
     - `feat1`: deepest temporal-A feature upsampled to `[B, 256, 256, 256]`
     - `feat2`: deepest temporal-B feature upsampled to `[B, 256, 256, 256]`
  7) Contrastive loss consumes `feat1`, `feat2`, and `label`.
- Note:
  - this is the `phi='b0'` path; `phi='b1'` uses `[64, 128, 320, 512]` stages instead.

### 2026-05-24 (Update 96)
- User asked the relation between `feat1/feat2` and `x1/x2` in SFEARNet.
- Confirmed relation from `model/sfearnet/__init__.py`:
  - `x1 = backbone(input1)` and `x2 = backbone(input2)` are lists of 4 feature maps each.
  - `feat1 = interpolate(x1[3], size=(H,W))`
  - `feat2 = interpolate(x2[3], size=(H,W))`
  - therefore, `feat1/feat2` are the deepest-stage features from `x1/x2` (index 3), upsampled to input resolution.

### 2026-05-24 (Update 97)
- User asked whether using only the 4th backbone layer for contrastive supervision is appropriate instead of using all backbone outputs together.
- Conclusion:
  - current design is reasonable and consistent with the code;
  - using `x1[3]` / `x2[3]` focuses contrastive supervision on the deepest semantic representation, which is better aligned with label-guided similarity/dissimilarity learning;
  - the four backbone stages have different resolutions, so they should not be mixed directly as one tensor for contrastive loss.
- Caveat:
  - if a future variant wants multi-scale contrastive supervision, it should define separate projections/losses per stage rather than collapsing the full list directly.

### 2026-05-24 (Update 98)
- User asked whether pseudo-change supervision should use the full `x1/x2` backbone outputs or only the 4th layer.
- Conclusion:
  - for the current contrastive branch, the 4th layer (`x1[3]`, `x2[3]`) remains the better choice;
  - pseudo-change suppression benefits more from deep semantic features than from shallow texture/detail features;
  - the full `x1/x2` lists should not be merged directly because their stage resolutions are inconsistent.
- Follow-up:
  - if multi-scale contrastive learning is needed later, it should be implemented as separate stage-wise branches with independent projections.

### 2026-05-24 (Update 99)
- User asked whether multi-scale contrastive learning would be better.
- Conclusion:
  - multi-scale contrastive can improve robustness in principle, but it is not automatically better than single deep-layer contrastive;
  - for pseudo-change suppression, a strong deep-layer main branch is usually more stable;
  - multi-scale contrastive should be treated as an ablation candidate, not the default replacement.
- Risk:
  - if shallow and deep features are fused without careful alignment, noise and resolution mismatch may hurt training.

### 2026-05-24 (Update 100)
- User explicitly requested that `PROJECT_PROGRESS.md` and `HANDOVER_PROMPT.txt` be maintained every turn without repeated reminders.
- Action:
  - future responses will treat the dual-file maintenance as a default requirement.
- No code changes in this round.

### 2026-05-24 (Update 101)
- User asked whether using only the 4th backbone layer is definitely better.
- Clarification:
  - this is not a guaranteed conclusion;
  - current recommendation is based on task structure and code consistency, not on a completed ablation result;
  - single deep-layer contrast is the safer default, while multi-scale contrast remains a candidate for later testing.
- Action item:
  - keep the current default design, and treat multi-scale contrast as an ablation rather than a replacement.

### 2026-05-24 (Update 102)
- User asked how to design the ablation study for contrastive learning.
- Recommended protocol:
  - keep backbone, optimizer, schedule, and random seed fixed;
  - change only the contrastive branch design;
  - compare no-contrastive, single deep-layer, single shallow-layer, and multi-scale variants.
- Implementation note:
  - multi-scale contrast should be computed stage-wise, after per-stage projection/alignment;
  - do not directly merge raw multi-stage features with different resolutions.
- Evaluation note:
  - report OA, F1, IoU, Precision, Recall;
  - add pseudo-change visual cases if the dataset does not provide explicit pseudo-change labels.

### 2026-05-24 (Update 103)
- User proposed a PFEM replacement: `PCSE` / `Pseudo-change Suppression Enhancement Module`.
- Current assessment:
  - the idea is directionally valid because it explicitly targets reliable-difference filtering rather than plain difference amplification;
  - the most useful part is the reliability gate `G_i`, which can suppress photometric/shadow/seasonal noise;
  - however, the module is not yet proven and can degenerate if the gate lacks supervision or if the similarity/context branches are too weak.
- Action item:
  - frame this as a candidate improvement and validate it with ablation against the original PFEM.

### 2026-05-24 (Update 104)
- User worried about modifying code without safe rollback.
- Provided a practical rollback workflow:
  1) create a dedicated experiment branch before edits;
  2) create a checkpoint commit (WIP allowed) before risky changes;
  3) for extra safety, export a patch backup file;
  4) if result is bad, return by switching back to checkpoint commit/branch.
- Collaboration rule:
  - follow this workflow by default for future structural edits to reduce rollback risk.

### 2026-05-24 (Update 105)
- User requested creating the snapshot branch immediately.
- Action completed:
  - created and switched to `backup/pre-pcse-20260524`.
- Next:
  - create a checkpoint commit on this branch before starting PCSE code edits.
