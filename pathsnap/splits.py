"""Frozen trip-grouped folds shared by every model and ablation.

Folds are created once by scripts/make_splits.py and saved to
outputs/splits/folds.json; everything else READS that file.
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_PATH = os.path.join(ROOT, "outputs", "splits", "folds.json")
SEED = 20261118
N_FOLDS = 5

# Training trips whose GPS track does not follow its own reference route
# (median aligned distance > 100 m; most fixes clustered near the start while
# the route runs for km). Derived labels are unreliable -> excluded from
# training loss and CV scoring; listed in the manifest. See outputs/eda/bad_alignment_trips.png
BAD_LABEL_TRIPS = {"00000033", "00000054"}


def load_folds() -> dict:
    with open(SPLIT_PATH) as f:
        return json.load(f)
