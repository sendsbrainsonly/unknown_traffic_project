# A-2 Formal Attempt v1 Failure

- Status: **FAILED — preserved diagnostic evidence**
- Stage reached: first formal training epoch; no epoch metric row or checkpoint was produced
- GPU selected at launch: physical GPU 5
- Strict leakage audit before launch: PASS
- Unknown samples loaded: 0
- Exception: `FloatingPointError: non-finite total loss: nan`
- Original combined execution log: `.tmux-task/stage3_primary_continue_v1/output.log`
- A-2 Final Test reached: no
- A-3 started: no

The smoke run remained finite but showed unstable loss magnitudes. The failed
formal configuration is preserved in this bundle and will not be overwritten.
