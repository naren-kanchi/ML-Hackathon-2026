"""Raw-file parsing for PathSnap trips.

Inference-side loaders (track, nodes, arcs) live here. The reference route is
loaded ONLY by `pathsnap.labels.load_route`, so the prediction path never has
to import route-reading code.

File schemas (verified against the 90 student-side trips, see outputs/eda):
  <id>.track : lon lat t      (float, float, float seconds), whitespace separated
  <id>.nodes : lon lat        (row index = node_id, 0-based)
  <id>.arcs  : from to        (int node ids, row index = arc_id, 0-based, directed)
Arc and node IDs are local to each trip.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Student folder")
PARTITIONS = ("train", "student_test")


def trip_dir(partition: str, trip_id: str, root: str = DATA_ROOT) -> str:
    return os.path.join(root, partition, trip_id)


def list_trips(partition: str, root: str = DATA_ROOT) -> list[str]:
    d = os.path.join(root, partition)
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def _load(path: str, dtype) -> np.ndarray:
    a = np.loadtxt(path, dtype=dtype, ndmin=2)
    return a


@dataclass
class Trip:
    """Everything the predictor is allowed to see. No route here by design."""
    trip_id: str
    track: np.ndarray            # (T, 3) lon, lat, t
    nodes: np.ndarray            # (N, 2) lon, lat
    arcs: np.ndarray             # (A, 2) from, to
    meta: dict = field(default_factory=dict)


def load_trip(partition: str, trip_id: str, root: str = DATA_ROOT) -> Trip:
    d = trip_dir(partition, trip_id, root)
    track = _load(os.path.join(d, f"{trip_id}.track"), np.float64)
    nodes = _load(os.path.join(d, f"{trip_id}.nodes"), np.float64)
    arcs = _load(os.path.join(d, f"{trip_id}.arcs"), np.int64)
    return Trip(trip_id, track, nodes, arcs, {"partition": partition})


def load_trip_from_files(track_path: str, nodes_path: str, arcs_path: str, trip_id: str = "adhoc") -> Trip:
    """Used by the API / demo: proves inference needs only these three files."""
    return Trip(trip_id, _load(track_path, np.float64), _load(nodes_path, np.float64), _load(arcs_path, np.int64))
