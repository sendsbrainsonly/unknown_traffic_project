# Archive metadata correction

2026-11-03, [agent:codex]. The first archive validation (`iscx_pr_archive_20261103`, exit 1) rejected 31 absolute external-artifact paths. The validator requires every artifact path to be relative, including external references.

The failed manifest is retained as `archive_failure_manifest.json`; logs and exit status remain in the project-local tmux directory. Only external-reference path syntax is corrected to the equivalent `../<source-path>`. No source, prediction, metric CSV, split, seed or training result is changed. Independent metric replay is rerun before refreshing and validating the corrected archive.
