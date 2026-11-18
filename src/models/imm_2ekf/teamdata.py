"""IMM (straight EKF + coordinated-turn EKF) features on the TEAM preprocessing.

Input: the outputs of
    python -m src.data.preprocess  <raw> data/processed
    python -m src.features.build_features data/processed data/features
i.e. cleaned/thinned GPS points per trip (points.parquet: x, y, t, segment, ...)
and one emission row per (point, candidate arc) with the team's 17 features.

For each trip we run the IMM separately on every GPS segment (the team splits at
gaps > 120 s; inside a segment the tracker additionally re-initialises after a
time step > reinit_gap_s). Its prior/posterior estimates become 12 extra
features per candidate row (see trackers.tracker_features).
Only x, y, t of the GPS points are used -> inference-safe.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data.preprocess import _read, load_trip
from src.features.build_features import EMISSION_FEATURES

from .trackers import TRACK_NAMES, TrackerParams, run_tracker, tracker_features

IMM_NAMES = ["imm_" + n for n in TRACK_NAMES] + ["imm_mu_turn"]


class _ArcGeom:
    """Minimal road geometry exposing what tracker_features needs."""

    def __init__(self, arcs: pd.DataFrame, nodes_xy: np.ndarray):
        n = int(arcs.index.max()) + 1
        self.p0 = np.zeros((n, 2))
        self.p1 = np.zeros((n, 2))
        ids = arcs.index.values
        self.p0[ids] = nodes_xy[arcs.u.values]
        self.p1[ids] = nodes_xy[arcs.v.values]
        d = self.p1 - self.p0
        self.heading = np.arctan2(d[:, 1], d[:, 0])

    def project_points(self, pts, arc_ids):
        a = self.p0[arc_ids]
        d = self.p1[arc_ids] - a
        L2 = np.maximum((d * d).sum(1), 1e-9)
        f = np.clip(((pts - a) * d).sum(1) / L2, 0.0, 1.0)
        foot = a + f[:, None] * d
        return np.hypot(*(pts - foot).T), f * np.sqrt(L2), f, foot


def run_imm_segments(pts: pd.DataFrame, p: TrackerParams, reinit_gap_s: float) -> dict:
    xy, t, seg = pts[["x", "y"]].values, pts.t.values, pts.segment.values
    parts = []
    for s in np.unique(seg):
        idx = np.nonzero(seg == s)[0]
        parts.append((idx, run_tracker(xy[idx], t[idx], "imm", p, reinit_gap_s)))
    out = {}
    for k in parts[0][1]:
        shape = (len(pts),) + parts[0][1][k].shape[1:]
        arr = np.zeros(shape, dtype=parts[0][1][k].dtype)
        for idx, o in parts:
            arr[idx] = o[k]
        out[k] = arr
    return out


def trip_table(processed_dir, features_dir, part, tid, p: TrackerParams, reinit_gap_s: float, with_imm=True):
    """Return (F, names, pt, arc, label, n_points, true_arc) for one trip."""
    trip = load_trip(processed_dir, part, tid)
    em = _read(os.path.join(features_dir, part, f"{tid}_emission"))
    em = em.sort_values(["pt", "dist_m"], kind="stable").reset_index(drop=True)
    base = em[EMISSION_FEATURES].astype(float).fillna(0.0).values.astype(np.float32)
    names = list(EMISSION_FEATURES)
    F = base
    if with_imm:
        tr = run_imm_segments(trip.points, p, reinit_gap_s)
        geom = _ArcGeom(trip.arcs, trip.nodes_xy)
        ctx = SimpleNamespace(cands=SimpleNamespace(obs=em.pt.values.astype(int), arc=em.arc_id.values.astype(int)),
                              rn=geom)
        F = np.hstack([base, tracker_features(ctx, tr, imm=True, sigma_z=p.sigma_z)])
        names += IMM_NAMES
    return dict(F=F, names=names, pt=em.pt.values.astype(int), arc=em.arc_id.values.astype(int),
                label=em.label.values.astype(int), n_points=len(trip.points),
                true_arc=trip.points.true_arc.values.astype(int))
