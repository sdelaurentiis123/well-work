# Paper draft status — ML4PS 2026 submission

## Scope lock

**Workshop paper is FNO-only.** Walrus comparison, Athena++ cross-framework work, and any other architecture/dataset axes are deferred to Paper 2 (journal-length, with APAM advisor collaboration). This decision was made after recognizing that in-distribution Walrus on MHD_64 is contaminated (Walrus's pretrain corpus includes MHD_64), while OOD Walrus-on-Athena++ without matching FNO-on-Athena++ creates an architecture-vs-framework confound we cannot resolve within the 6-page workshop-paper scope.

The workshop paper's three contributions stand alone without Walrus:
1. Data-efficiency advantage of MHD→MHD pretraining (Fig. 2).
2. Physics-specificity proof via the non-MHD control (Fig. 1).
3. Detailed audit of what transfers vs what doesn't (Figs. 3–6).

## What's in place

- `main.tex` — complete draft, all six figures referenced, all numerical claims sourced from committed artifacts.
- `references.bib` — 11 citations covering Polymathic models, Well dataset, FNO, Goldreich-Sridhar, plasma ML prior art.
- `figures/` — 6 PDFs + 6 PNGs, consistent styling.
- `make_figures.py` — regenerate figures from `p1/` artifacts; no compute.

## Known gaps in the draft

1. **NeurIPS template.** I'm using `neurips_2024.sty` as a placeholder. Swap to the ML4PS 2026 workshop template when it's released (they typically follow the NeurIPS main-track template with minor modifications).
2. **Author block.** Only Stan listed. Add co-authors once advisor structure for the workshop submission is decided (Sironi? Zhu? Hansen? CC Cranmer / McCabe if they collaborate on the direction given Polymathic involvement).
3. **Related work section (§2).** Currently terse. Expand if length allows, especially: (a) more specific citation of MPP's transfer experiments vs ours; (b) named-citation for Zhu's DivControlNN when its arxiv ID is stable; (c) if Walrus paper is accepted/published by submission time, update citation.
4. **Methods section hyperparameter details.** A supplementary appendix is warranted listing all sweep configurations and per-seed numbers. Current `hp_summary.json` in repo root is the source-of-truth; I'll add an appendix table in a v2 pass.
5. **Section 5 discussion.** Could be sharper. Three claims are stated but "actionable implications" sometimes over-reach given the single-architecture / single-dataset setup. Soften or scope as we iterate.

## Per-figure verification

| Figure | Source artifact | Numbers cross-checked |
|---|---|---|
| fig1 headline | `p1/hp_summary.json`, `p1/runs/ns_ft_01_s{0,1,2}/log.jsonl` | scratch 0.419±0.008, NS 0.515±0.002, MHD 0.301±0.000 ✓ |
| fig2 data-eff | `p1/runs/*/log.jsonl` + `hp_summary.json` for 1% row | 100% pair 0.272/0.268, 10% pair 0.299/0.277, 1% pair 0.419/0.301 ✓ |
| fig3 cascade | `p1/evals/physics/{baseline_01,baseline,ft_01}/aniso_step1.npz` | baseline_01 falls off by ~2 dex ≥ k_⊥=8 ✓ |
| fig4 rollout | `p1/evals2/{baseline,baseline_01,ft_01}/results.json` | ft_01 step50 ≈ 15.7 vs baseline_01 ≈ 2.0 ✓ |
| fig5 equipart | `p1/evals/physics/{baseline,baseline_01,ft_01}/conservation.npz` | ft_01 drifts below 0.5 by step 30 ✓ |
| fig6 conserv | same | ft_01 divB reaches ~0.15 vs baseline_01 ~0.04 ✓ |

## Checklist before submission

- [ ] Swap in ML4PS 2026 template (when released)
- [ ] Authors block finalized
- [ ] Related-work expansion pass
- [ ] Supplementary appendix with HP sweep tables
- [ ] Read by an independent physicist (Sironi) to catch physics-jargon missteps
- [ ] Read by an independent ML reviewer to catch ML-jargon missteps
- [ ] One more pass on Section 5 to scope claims tightly to what was measured
- [ ] Proofread
- [ ] Verify all citations render, no broken refs
- [ ] 6-page limit check after template swap

## File layout inside `p1/paper/`

```
p1/paper/
├── main.tex              # LaTeX source, complete draft
├── references.bib        # 11 citations
├── DRAFT_STATUS.md       # this file
├── make_figures.py       # regenerate figures from p1/ artifacts
└── figures/
    ├── fig1_headline_physics_specificity.{pdf,png}
    ├── fig2_data_efficiency.{pdf,png}
    ├── fig3_cascade.{pdf,png}
    ├── fig4_long_horizon_failure.{pdf,png}
    ├── fig5_equipartition_drift.{pdf,png}
    └── fig6_conservation_violation.{pdf,png}
```

## How to build

```bash
cd p1/paper/
# if you don't have a LaTeX distribution:
#   brew install --cask mactex  (or whatever's appropriate)
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Figures are committed as PDFs so the build is self-contained. To regenerate them from source artifacts (e.g., after adding new eval data):

```bash
cd /path/to/plasma-tinkering
source mlvenv/bin/activate
python p1/paper/make_figures.py
```
