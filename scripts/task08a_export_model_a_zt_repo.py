# -*- coding: utf-8 -*-
"""Repository-local launcher for the existing Model A z_t exporter.

The original Task 08 script imports ``uer`` before parsing ``--tf-code-dir``
and therefore cannot discover a repository-local TrafficFormer checkout on its
own. This additive launcher makes the pinned source under ``tf_runtime/code``
importable, supplies that directory as the default ``--tf-code-dir``, and then
executes the original script unchanged.
"""

from pathlib import Path
import runpy
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TF_CODE_DIR = PROJECT_ROOT / "tf_runtime" / "code"
TASK08 = PROJECT_ROOT / "scripts" / "task08_export_model_a_zt.py"


def main():
    required = [
        TF_CODE_DIR / "uer" / "opts.py",
        TF_CODE_DIR / "fine-tuning" / "run_classifier.py",
        TF_CODE_DIR / "models" / "encryptd_vocab.txt",
        TF_CODE_DIR / "models" / "bert" / "base_config.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "repository-local TrafficFormer/UER source is incomplete:\n  "
            + "\n  ".join(missing)
        )

    sys.path.insert(0, str(TF_CODE_DIR))
    if not any(
        arg == "--tf-code-dir" or arg.startswith("--tf-code-dir=")
        for arg in sys.argv[1:]
    ):
        sys.argv.extend(["--tf-code-dir", str(TF_CODE_DIR)])
    runpy.run_path(str(TASK08), run_name="__main__")


if __name__ == "__main__":
    main()
