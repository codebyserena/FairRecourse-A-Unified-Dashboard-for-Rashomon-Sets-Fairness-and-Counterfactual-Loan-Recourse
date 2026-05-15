"""Run the FairRecourse pipeline from a single command.

This keeps the existing scripts independent while making the full experiment
reproducible from the project root:

    python3 run_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINE_STEPS = [
    "src/data.py",
    "src/train_models.py",
    "src/rashomon.py",
    "src/fairness.py",
    "src/recourse.py",
    "src/explainability.py",
]


def run_step(script: str) -> None:
    print(f"\n=== Running {script} ===", flush=True)
    subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    for script in PIPELINE_STEPS:
        run_step(script)
    print("\nPipeline complete. Start the dashboard with: python3 app.py")


if __name__ == "__main__":
    main()
