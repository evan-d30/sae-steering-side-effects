#!/usr/bin/env python
"""Quickly summarize model result JSON files in results/.

Usage:
    python scripts/summarize_results.py results/
"""

from pathlib import Path
import json
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")

for path in sorted(root.rglob("*summary*.json")):
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(f"{path}: could not read JSON: {exc}")
        continue

    print("=" * 80)
    print(path)
    for key in [
        "model",
        "n_features",
        "n_contexts",
        "sae_release_actual",
        "sae_id_actual",
        "downstream_sae_id_actual",
        "mean_full_no_magnitude_minus_freq",
        "mean_full_no_magnitude_minus_activation_magnitude",
    ]:
        if key in data:
            print(f"{key}: {data[key]}")
