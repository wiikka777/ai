#!/usr/bin/env python3
"""
Compatibility launcher for the improved comparison experiment.

Some environments invoke:
    python scripts/experiments/comparison_experiment_v2.py

while the maintained implementation lives at repository root:
    comparison_experiment_v2.py
"""

import os
import runpy


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(current_dir))
    target = os.path.join(repo_root, "comparison_experiment_v2.py")
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
