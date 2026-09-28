# Stage42-S Python cache drift during Stage42-T reuse

The Stage42-T runner imported the frozen Stage42-S extraction and replay modules. Python regenerated two `__pycache__` files in the Stage42-S bundle. A fresh Stage42-S bundle validation detected only these two hash mismatches; source scripts, protocol, checkpoints, and metric files passed their existing hashes.

The Stage42-S manifest SHA256 before refreshing its inventory was `5abdce63e916107df5b6ec78668bced7d5541c9ecff2d21ea5b952385b1b8a7b`. The preserved failure log is `.tmux-task/stage42s-final-revalidate-20260928/output.log` under the project root. The prior and actual cache hashes were:

| Cache file | Previous manifest SHA256 | Regenerated SHA256 |
|---|---|---|
| `__pycache__/build_unknown_cache.cpython-310.pyc` | `aadb342d4718a17219bc0c3c03f9046e5529f833c4cf23c1988abfe66f77a1ff` | `005213d4d0cfa1345b39f43a3b877cda4136074370fc06277697f5f381e3edbe` |
| `__pycache__/verify_candidate.cpython-310.pyc` | `1e566e0e94e77054eed93d9dcd47d7216d85dfdfaa2e77481b74766c5d509823` | `6a16e821f0548f5874e5cf60f1a65a352bb9d799e0b4d5de7e99d7d0491d4a10` |

The relevant Stage42-S source hashes stayed `build_unknown_cache.py=669d999018bbb51d66fb17ac75c65230768d2e079860f3e330c277d4865e1943` and `verify_candidate.py=ab2e728ecb940bd47b71efbe727315bf36a1c809a7ebcff451b9c8d1d8aed386`, matching the pre-feature freeze in `candidate_protocol.json`. The Stage42-T runner now sets `sys.dont_write_bytecode=True` before importing Stage42-S modules. The two caches are non-scientific derived files; Stage42-S source/protocol/scoring outputs were not rewritten. The Stage42-S artifact manifest was refreshed only to record their current bytes, then hash-validated.
