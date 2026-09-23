#!/usr/bin/env python3
"""Diagnose BLAS-thread sensitivity of the historical randomized PCA replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from task10_stage2_audit import _load_aligned, _standardize  # noqa: E402
from task10f_covariance_diagnosis import fit_pca_train_only  # noqa: E402


def main() -> None:
    zt_dir = PROJECT_ROOT / "outputs/stage1/modelA/embeddings"
    zg_dir = PROJECT_ROOT / "outputs/stage1/modelB/seeds/seed2"
    zt, labels, zg = _load_aligned(zt_dir, zg_dir)
    _, zf = _standardize(zt, zg)
    zf = {split: value.astype(np.float32, copy=False) for split, value in zf.items()}
    ftp_id = json.loads(
        (PROJECT_ROOT / "data/fig_graph/all_flows/label_map.json").read_text()
    )["class_to_id"]["FTP"]
    train_mask = labels["train"] == ftp_id
    val_mask = labels["val"] == ftp_id
    for limit in (1, 2, 4, 8, 16, 32):
        with threadpool_limits(limits=limit):
            pca, train, val = fit_pca_train_only(zf["train"], zf["val"], 64, seed=0)
            model = GaussianMixture(
                n_components=2,
                covariance_type="full",
                reg_covar=1e-3,
                n_init=3,
                max_iter=200,
                random_state=0,
            ).fit(train[train_mask])
            print(
                json.dumps(
                    {
                        "thread_limit": limit,
                        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
                        "train_nll": float(-model.score(train[train_mask])),
                        "validation_nll": float(-model.score(val[val_mask])),
                        "weights": [float(item) for item in model.weights_],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
