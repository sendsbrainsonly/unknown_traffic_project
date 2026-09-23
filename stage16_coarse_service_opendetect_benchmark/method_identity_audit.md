# Stage 16 method identity audit

## Gate outcome

`METHOD_IDENTITY_NOT_DISTINCT` — the requested `OURS-Service6` versus
`OpenDetect-Service6` claim is not identifiable from the frozen DQ-3F artifacts.
No new training was launched.

## Direct code provenance

- DQ-3F `train_dq3f.py` lines 47--56 add `Projects/Open-Detect/code` and
  `Projects/Open-Detect/reproduction` to `sys.path`, then import
  `CorrectedOpenDetectNet`, `reset_prototypes_in_place`, `run_epoch`, and
  Open-Detect's `weight_init`.
- Lines 130--143 instantiate `CorrectedOpenDetectNet` and use Adam plus the
  Open-Detect reproduction training helper.
- Lines 179--193 execute Open-Detect's loss/training loop, in-place prototype
  reset, nearest-prototype validation, and Known-Validation Accuracy checkpoint
  selection.
- The frozen DQ-3F config explicitly records
  `training_protocol = corrected-paper`.
- `CorrectedOpenDetectNet` subclasses the released `OpenDetectNet`. Its documented
  changes are a connected final decoder residual block, bounded log-variance,
  paper-consistent KL/prototype classification, and removal of the released
  constant entropy term. These are Open-Detect reproduction corrections, not an
  independent project model.

## What is and is not distinct

| Item | DQ-3F Service model | Released Open-Detect | Identity implication |
|---|---|---|---|
| Core class | `CorrectedOpenDetectNet(OpenDetectNet)` | `OpenDetectNet` | Same method family and inherited architecture |
| Input | 32x32 one-channel byte image | 32x32 one-channel byte image in local reproduction | Same representation family |
| Encoder | ResNet18 VAE encoder | ResNet18 VAE encoder | Same |
| Prototype rule | learned class prototypes | learned class prototypes | Same mechanism |
| Objective | corrected paper-consistent Open-Detect loss | released-code Open-Detect loss | Reproduction variant difference |
| Training helper | Open-Detect `run_reproduction.py` | Open-Detect code/reproduction | Same provenance |
| Service adaptation | six output prototypes | six output prototypes required | Task-only adaptation |

The defensible comparison would be **Corrected Open-Detect versus released-code
Open-Detect**, not **our method versus Open-Detect**. Renaming the corrected
variant as an independent method would violate the Stage 16 instruction not to
manufacture method differences.

The project does contain a separately named Stage 14C-6 F2 own-method pipeline,
but it was not used in DQ-3F and cannot be silently substituted while claiming
to reuse the DQ-3F method.

## Source SHA256

- DQ-3F trainer: `cc0efd8107fa8010415d4f64e6779c2f238c87a835db445b5d5fe55e5ccef89e`
- Open-Detect corrected model: `b904447ae41a3144e2408187aa200b0796643cffb1c03ea3402dd9718820dfbd`
- Open-Detect reproduction runner: `7d674a1cb135a8762e3818538449a0e88b103a2943e603ce860bddc6fd752491`
- Open-Detect released model: `756fc2d3c5ac146ce7fb1f0e1fa00a07b7ca7b626b2baffc2e14562121c1c636`

## Decision

The method-identity gate fails before GPU work. Stage 16-2 through Stage 16-5
are `NOT_RUN_IDENTITY_GATE`; the existing DQ-3F/DQ-4 measurements are retained
below only as historical corrected-Open-Detect evidence, never as a two-method
comparison.
