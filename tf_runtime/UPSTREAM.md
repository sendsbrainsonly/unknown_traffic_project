# TrafficFormer / UER source provenance

This directory vendors the source files required by Model A from a fixed
upstream snapshot.

- Upstream: <https://github.com/IDP-code/TrafficFormer>
- Commit: `6d0ba64d82e74fb130c6c7301ef20885dbfbdf29`
- Upstream commit date: 2025-01-18
- License: MIT; the upstream license is retained at `code/LICENSE`.
- Local source root: `tf_runtime/code`

Upstream-tracked `__pycache__/*.pyc` files are intentionally excluded because
they are interpreter-specific compiled caches rather than source. The upstream
Windows-only `data_generation/SplitCap.exe` binary is also excluded: this
project uses its own audited Stage 0 flow splitter and does not execute that
binary. Apart from these exclusions, files under `code/` are copied from the
pinned commit without local edits. `SOURCE_MANIFEST.sha256` records the imported
file hashes.

The snapshot includes TrafficFormer fine-tuning code, its bundled `uer` package,
`models/encryptd_vocab.txt`, and `models/bert/base_config.json`. It does not
include any pretrained or fine-tuned model weights; the selected upstream commit
does not track those files. A locally downloaded official pretrained model may
exist at `code/models/pretrained_model.bin`; it remains ignored by Git and its
provenance is recorded in `PRETRAINED_MODEL.md`.

Do not install `tf_runtime/code/requirements.txt` blindly into the shared
TrafficClassifier environment. Its pins describe the upstream environment and
may conflict with the workspace environment. Audit missing imports first and
request authorization before changing the shared environment.

Use `scripts/task08a_export_model_a_zt_repo.py` to run the existing Model A
embedding exporter with this repository-local source tree.
