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

from pathsnap.candidates import Candidates
from pathsnap.context import TripContext
from pathsnap.hmm import HMMParams, classifier_only, reconstruct_route, viterbi
from pathsnap.io import Trip
from pathsnap.io import load_trip as ps_load_trip
from pathsnap.labels import load_route
from pathsnap.metrics import continuity, macro_f1_trip, route_metrics
from pathsnap.roadnet import RoadNet

from .trackers import TRACK_NAMES, TrackerParams, run_tracker, tracker_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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


def ctx_from_candidate_lists(trip: Trip, rn: RoadNet, cand_arcs: list, probs: list):
    """TripContext whose candidates are an arbitrary per-fix arc list (duplicates
    mapped to canonical ids; probabilities of duplicate arcs are summed)."""
    xy = rn.project_track(trip.track)
    obs, arc, pr = [], [], []
    for t, (a, p) in enumerate(zip(cand_arcs, probs)):
        ca = rn.canon[np.asarray(a, int)]
        u, inv = np.unique(ca, return_inverse=True)
        ps = np.bincount(inv, weights=np.asarray(p, float), minlength=len(u))
        obs.append(np.full(len(u), t)); arc.append(u); pr.append(ps)
    obs, arc, pr = np.concatenate(obs), np.concatenate(arc), np.concatenate(pr)
    dist, off, frac, foot = rn.project_points(xy[obs], arc)
    o = np.lexsort((arc, dist, obs))
    obs, arc, pr, dist, off, frac, foot = obs[o], arc[o], pr[o], dist[o], off[o], frac[o], foot[o]
    ptr = np.searchsorted(obs, np.arange(len(xy) + 1))
    c = Candidates(obs, arc, dist, off, frac, foot, ptr, radius=np.nan, max_k=-1)
    return TripContext(trip, rn, xy, trip.track[:, 2].astype(float), c), pr


def decode_team_trip(part, tid, probs=None, table=None, hp=None):
    """Shared HMM/Viterbi + metrics on the team's cleaned points for one trip.
    probs: flat per-row probabilities in emission order (sorted by pt, dist_m), or
    table: DataFrame(pt, arc_id, proba) matched by (pt, arc_id). Labels = team true_arc
    (denominator = labelled points). hp: pathsnap.hmm.HMMParams (default if None)."""
    tp = load_trip(os.path.join(ROOT, "data", "processed"), part, tid)
    em = _read(os.path.join(ROOT, "data", "features", part, f"{tid}_emission"))
    em = em.sort_values(["pt", "dist_m"], kind="stable").reset_index(drop=True)
    if table is None:
        p_flat = np.asarray(probs)
    else:   # match by (pt, arc_id), not by row order
        p_flat = em[["pt", "arc_id"]].merge(table, on=["pt", "arc_id"], how="left").proba.fillna(0.0).values
    raw = ps_load_trip(part, tid)
    pts = tp.points
    trip = Trip(tid, np.column_stack([pts.lon.values, pts.lat.values, pts.t.values]), raw.nodes, raw.arcs)
    rn = RoadNet(raw)
    n = len(pts)
    ptr = np.searchsorted(em.pt.values, np.arange(n + 1))
    lists = [em.arc_id.values[ptr[i]:ptr[i + 1]] for i in range(n)]
    plist = [p_flat[ptr[i]:ptr[i + 1]] for i in range(n)]
    ctx, pr = ctx_from_candidate_lists(trip, rn, lists, plist)
    lab = pts.true_arc.values.astype(int)
    lab = np.where(lab >= 0, rn.canon[np.maximum(lab, 0)], -1)
    L = lab >= 0
    route_c = rn.canon[load_route(part, tid)]
    rp = pts.route_pos.values[L]
    covered = (int(rp.min()), int(rp.max())) if L.any() else None
    hp = hp or HMMParams()
    out = []
    for mode in ("clf", "hmm"):
        dec = classifier_only(ctx, pr) if mode == "clf" else viterbi(ctx, pr, hp)
        route, breaks = reconstruct_route(ctx, dec, hp)
        a = dec["arc"]
        r = dict(partition=part, trip=tid, mode=mode, n_fix=int(L.sum()), top1=float((a[L] == lab[L]).mean()),
                 macro_f1=macro_f1_trip(a[L], lab[L]), continuity=continuity(ctx, a, hp, dec["netdist"]),
                 restarts=sum(1 for _, x in dec["restarts"] if x != "start"), route_breaks=breaks,
                 top1_clean=np.nan, log_loss=np.nan, log_loss_covered=np.nan, brier=np.nan, miss_rate=np.nan)
        r.update(route_metrics(rn, route, route_c, covered))
        out.append(r)
    return out
