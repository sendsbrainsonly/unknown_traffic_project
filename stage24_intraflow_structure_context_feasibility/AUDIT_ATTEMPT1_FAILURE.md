# Stage 24 audit attempt 1 — preserved failure

The first preflight stopped before producing any dataset summary because it assumed every source provenance `packet_refs` timestamp sequence was monotone. A Known Train/Validation flow violated that assumption (`f81103535cc0e413410f025eadc1d658894cfd5b2dba1ba5e12f0b16b378996c`). The full traceback and nonzero exit status remain at `.tmux-task/codex_stage24_dataset_audit_v1_20260924/` under the project root.

This is an input-order audit issue, not permission to rewrite source data or the frozen Stage 21 graph cache. Follow-up preflight retains the original packet order, counts all such flows explicitly, and uses min/max timestamps only for a separate read-only causal-context inventory. No model training or Test feature access occurred in the failed attempt.
