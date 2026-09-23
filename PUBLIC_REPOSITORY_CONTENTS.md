# Public Repository Contents

This repository is a reproducibility and evidence index, not a copy of the complete local experiment workspace.

## Included in ordinary Git

- implementation source, command-line scripts and tests;
- frozen configurations and protocol definitions;
- method and dataset documentation;
- stage-level `README.md` and `RESULTS.md` files;
- stage-level `manifest.json`, completion verification and small summary metadata;
- selected small aggregate metrics needed to support claims in the documentation.

## Kept project-local

- raw and derived datasets;
- model checkpoints and pretrained weights;
- embeddings, latent arrays and prediction dumps;
- packet/image caches and reconstructed views;
- complete per-run directories, optimizer histories and launcher logs;
- failed/interrupted binary artifacts;
- tmux runtime logs and mutable queue/progress files.

These files are not deleted. Their paths, hashes and roles remain recorded in stage manifests and result documents where available.

## Model release policy

No checkpoint is designated as the final project model at this snapshot because Stage 22 is still running. Once a final model is selected, publish only the minimum reproducibility set through Git LFS or a versioned GitHub Release, together with:

- exact experiment ID and configuration;
- source commit;
- SHA256 and byte size;
- dataset/protocol identifier;
- selection criterion and seed;
- license and upstream-weight provenance.

Do not place checkpoints directly in ordinary Git history.

## Current size boundary

At the 2026-09-23 audit, the local workspace occupied about 54 GiB on disk and had about 65 GiB of logical file content. Human-readable documents were about 1.26 MiB and source/scripts/configuration about 4.62 MiB. The size difference is experimental evidence and cache data, not project documentation.

The `.gitignore` intentionally excludes large data and artifact formats. Small ignored metric tables may be force-added only after checking that they are stable, necessary to support a documented result and comfortably below repository size limits.
