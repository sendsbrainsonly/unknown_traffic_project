# Visual review — accepted

- Date: 2026-10-16, Codex chat `01a144b7-de8c-7521-a7ae-d8575754129d`.
- Method: viewed generated PNGs with the local image viewer; checked labels, clipping, legend placement, panel alignment, numerical annotations, scales and color contrast. Numeric checks are recorded separately in `verification.json`.
- Figure 4-1: all seven methods and both datasets visible; error bars and mean annotations separate. Final PNG is byte-identical to the individually reviewed render-v2 image.
- Figure 4-2: labels and all four values per dataset readable; tasks explicitly state four/five classes. Final PNG is byte-identical to reviewed render-v2.
- Figure 4-3: four metrics, four held-out services, legend and hatch distinction readable; all bars start at zero and Known FRR uses its stated smaller scale. Final PNG is byte-identical to reviewed render-v2.
- Figure 4-4: directly viewed final render-v4; all 48 cells and separate color scales visible, no clipped tick labels or colorbar collisions. Darker colors consistently indicate better metric values. SVG/PDF structure verified as vector without embedded images.
- Figure 4-5: directly viewed final render-v4; attack labels, AUROC pairs, sample counts and UFAR bars align row by row. Tiny UFAR values remain numerically annotated; Bot/DDoS remain visible. Slowloris label is 0.04, consistent with manuscript rounding.
- Figure 4-6: all five prevalence points and three stated composition examples visible; legends clear. AUROC/AUPRC panel starts at 80%, explicitly disclosed in the caption. Final PNG is byte-identical to reviewed render-v2.
- Figure 4-7: all three configurations and metrics visible; rotated bar labels, sample counts and shared legend fit within image. Final PNG is byte-identical to reviewed render-v2.
- Manuscript: seven images/captions and references inserted in corresponding sections; image paths and numbering checked mechanically. No claim is made that a complete journal page-layout PDF has been typeset or reviewed.
- Decision: seven final plots accepted for use in the Markdown manuscript and subsequent paper typesetting.
