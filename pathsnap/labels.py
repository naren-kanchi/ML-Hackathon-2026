"""Reference-route access. Training-label and evaluation code ONLY.

Nothing in pathsnap/predict*, pathsnap/candidates.py, pathsnap/features*.py or
pathsnap/hmm.py may import this module.
"""
from __future__ import annotations

import os

import numpy as np

from .io import DATA_ROOT, trip_dir


def load_route(partition: str, trip_id: str, root: str = DATA_ROOT) -> np.ndarray:
    """Ordered list of directed arc ids traversed by the vehicle (trip-local)."""
    p = os.path.join(trip_dir(partition, trip_id, root), f"{trip_id}.route")
    return np.loadtxt(p, dtype=np.int64, ndmin=1)
