# Pilot continuation rule timing

The Stage 24 experiment matrix, frozen-input rules, and formal five-seed gate were recorded before any Stage 24 model result. The conservative **two-seed pilot continuation rule** in `scripts/aggregate_pilot.py` was specified after the first S1 / ISCX-VPN / seed-2022 result was visible (Macro-F1 0.851384, below E1 0.853407). It was specified before aggregation of the complete 16-run pilot. Therefore, this rule is an **amendment**, not a fully pre-result preregistered gate.

The rule permits five-seed confirmation only if one candidate beats both frozen baselines in mean Known-Validation Macro-F1 on each dataset and in at least three of four paired cells. It does not select by Known Test or Unknown data. The first visible S1 result was negative, but that fact does not make the amended rule independent of all pilot data. We report the full pilot table regardless of the rule outcome and make no formal statistical claim from this two-seed screen.
