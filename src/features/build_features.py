"""
PathSnap - Feature extraction and preparation for HMM map matching.

An HMM map matcher needs two kinds of scores, so we build two feature tables:

  EMISSION table   one row per (GPS point, candidate arc)
                   "how well does this GPS point fit this road?"
  TRANSITION table one row per (candidate at point i, candidate at point i+1)
                   "how plausible is driving from road A to road B in this time?"

Labels come from the reference route (via preprocess.align_to_route):
  emission   label = 1 if the candidate is the true arc, 0 if not, -1 if unknown
  transition label = 1 if both candidates are the true arcs,  0 if not, -1 if unknown
Rows with label -1 (parked off-network, beyond route coverage) are kept for
inference but must be excluded from training and evaluation.

The same code runs at inference time: for unseen trips the labels are all -1.

Usage (from the repo root):
    python -m src.features.build_features data/processed data/features
"""
from __future__ import annotations

import heapq
import json
import os
import sys
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.preprocess import Trip, _read, _write, list_trips, load_trip


@dataclass
class FeatureConfig:
    radius_m: float = 50.0          # candidate search radius around each GPS point
    max_candidates: int = 8         # keep at most K nearest candidate arcs per point
    densify_m: float = 5.0          # spacing of the KD-tree sample points along arcs
    route_cutoff_min_m: float = 300.0
    route_cutoff_factor: float = 3.0  # Dijkstra stops at max(min, factor*gc + 2*radius)
    backtrack_tol_m: float = 10.0   # small backwards moves on the same arc count as 0 m (GPS noise)


EMISSION_FEATURES = [
    "dist_m", "dist_rank", "dist_minus_min_m", "n_candidates",
    "heading_diff_deg", "heading_valid", "speed_mps", "proj_frac",
    "arc_length_m", "bidirectional", "to_out_degree", "from_in_degree", "dead_end",
    "stop_s", "is_stop", "dt_prev_s", "segment_start",
]
TRANSITION_FEATURES = [
    "gc_dist_m", "route_dist_m", "abs_diff_m", "route_gc_ratio", "dt_s", "route_speed_kmh",
    "unreachable", "same_arc", "u_turn", "bearing_change_deg",
]


# ----------------------------------------------------------------------------
# candidate generation + emission features
# ----------------------------------------------------------------------------
def _angle_diff(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 360
    return np.minimum(d, 360 - d)


def generate_candidates(trip: Trip, cfg: FeatureConfig) -> pd.DataFrame:
    pts, arcs, N = trip.points, trip.arcs, trip.nodes_xy
    P = pts[["x", "y"]].values
    # restrict the network to arcs near the trip, then sample points along each arc
    lo, hi = P.min(0) - cfg.radius_m - 200, P.max(0) + cfg.radius_m + 200
    A_all, B_all = N[arcs.u.values], N[arcs.v.values]
    near = (((A_all >= lo) & (A_all <= hi)).all(1) | ((B_all >= lo) & (B_all <= hi)).all(1))
    arc_ids = arcs.index.values[near]
    A, B = A_all[near], B_all[near]
    L = np.hypot(*(B - A).T)
    n = np.maximum(np.ceil(L / cfg.densify_m).astype(int), 1)
    owner = np.repeat(np.arange(len(A)), n + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
    samples = A[owner] + frac[:, None] * (B - A)[owner]
    tree = cKDTree(samples)

    hits = tree.query_ball_point(P, r=cfg.radius_m + cfg.densify_m / 2)
    pi = np.concatenate([np.full(len(h), i) for i, h in enumerate(hits)]).astype(np.int64)
    ai = owner[np.concatenate(hits).astype(np.int64)] if len(pi) else np.array([], np.int64)
    pair = np.unique(np.column_stack([pi, ai]), axis=0)
    pi, ai = pair[:, 0], pair[:, 1]

    # exact point-to-segment distance and projection fraction
    a, b, p = A[ai], B[ai], P[pi]
    ab = b - a
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-12), 0, 1)
    dist = np.hypot(*(p - (a + t[:, None] * ab)).T)
    c = pd.DataFrame({"pt": pi, "arc_id": arc_ids[ai], "dist_m": dist, "proj_frac": t})
    c = c[c.dist_m <= cfg.radius_m]
    c = c.sort_values(["pt", "dist_m"])
    c["dist_rank"] = c.groupby("pt").cumcount()
    c = c[c.dist_rank < cfg.max_candidates].reset_index(drop=True)

    # emission features
    g = c.groupby("pt").dist_m
    c["dist_minus_min_m"] = c.dist_m - g.transform("min")
    c["n_candidates"] = g.transform("size")
    ar = arcs.loc[c.arc_id.values]
    pp = pts.iloc[c.pt.values]
    c["heading_diff_deg"] = _angle_diff(pp.heading_deg.values, ar.bearing_deg.values)
    c["heading_valid"] = pp.heading_deg.notna().values.astype(int)
    c["speed_mps"] = pp.speed_prev_mps.values
    c["arc_length_m"] = ar.length_m.values
    for col in ["bidirectional", "to_out_degree", "from_in_degree", "dead_end"]:
        c[col] = ar[col].values.astype(float)
    c["stop_s"] = pp.stop_s.values
    c["is_stop"] = (pp.stop_s.values > 0).astype(int)
    c["dt_prev_s"] = pp.dt_prev.values
    c["segment_start"] = pp.dt_prev.isna().values.astype(int)
    c["segment"] = pp.segment.values
    c["t"] = pp.t.values
    true_arc = pp.true_arc.values
    c["label"] = np.where(true_arc < 0, -1, (c.arc_id.values == true_arc).astype(int))
    c.insert(0, "trip", trip.trip_id)
    return c


# ----------------------------------------------------------------------------
# network shortest paths (bounded Dijkstra with a cache)
# ----------------------------------------------------------------------------
class Router:
    def __init__(self, arcs: pd.DataFrame, n_nodes: int):
        order = np.argsort(arcs.u.values, kind="stable")
        self.v = arcs.v.values[order]
        self.w = arcs.length_m.values[order]
        self.ptr = np.r_[0, np.cumsum(np.bincount(arcs.u.values, minlength=n_nodes))]
        self.cache: dict[int, tuple[float, dict]] = {}

    def from_node(self, s: int, cutoff: float) -> dict:
        hit = self.cache.get(s)
        if hit is not None and hit[0] >= cutoff:
            return hit[1]
        dist = {s: 0.0}
        heap = [(0.0, s)]
        v, w, ptr = self.v, self.w, self.ptr
        while heap:
            d, x = heapq.heappop(heap)
            if d > dist.get(x, np.inf) or d > cutoff:
                continue
            for k in range(ptr[x], ptr[x + 1]):
                y, nd = int(v[k]), d + w[k]
                if nd < dist.get(y, np.inf) and nd <= cutoff:
                    dist[y] = nd
                    heapq.heappush(heap, (nd, y))
        if len(self.cache) > 20000:
            self.cache.clear()
        self.cache[s] = (cutoff, dist)
        return dist


# ----------------------------------------------------------------------------
# transition features
# ----------------------------------------------------------------------------
def transition_features(trip: Trip, cand: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    arcs, pts = trip.arcs, trip.points
    router = Router(arcs, len(trip.nodes_xy))
    by_pt = {p: g for p, g in cand.groupby("pt")}
    xy, t, seg = pts[["x", "y"]].values, pts.t.values, pts.segment.values
    true_arc = pts.true_arc.values
    out = []
    for i in range(len(pts) - 1):
        if seg[i] != seg[i + 1] or i not in by_pt or (i + 1) not in by_pt:
            continue
        ca, cb = by_pt[i], by_pt[i + 1]
        gc = float(np.hypot(*(xy[i + 1] - xy[i])))
        dt = float(t[i + 1] - t[i])
        cutoff = max(cfg.route_cutoff_min_m, cfg.route_cutoff_factor * gc + 2 * cfg.radius_m)
        a_id, a_t = ca.arc_id.values, ca.proj_frac.values
        b_id, b_t = cb.arc_id.values, cb.proj_frac.values
        a_len, a_v = arcs.length_m.loc[a_id].values, arcs.v.loc[a_id].values
        a_rev = arcs.reverse_arc_id.loc[a_id].values
        a_brg = arcs.bearing_deg.loc[a_id].values
        b_len, b_u = arcs.length_m.loc[b_id].values, arcs.u.loc[b_id].values
        b_brg = arcs.bearing_deg.loc[b_id].values
        R = np.full((len(a_id), len(b_id)), np.inf)
        for j in range(len(a_id)):
            dmap = router.from_node(int(a_v[j]), cutoff)
            rest_a = (1 - a_t[j]) * a_len[j]
            for k in range(len(b_id)):
                if a_id[j] == b_id[k] and (b_t[k] - a_t[j]) * a_len[j] >= -cfg.backtrack_tol_m:
                    R[j, k] = max(0.0, (b_t[k] - a_t[j]) * a_len[j])
                else:
                    sp = dmap.get(int(b_u[k]))
                    if sp is not None:
                        R[j, k] = rest_a + sp + b_t[k] * b_len[k]
        J, K = np.meshgrid(np.arange(len(a_id)), np.arange(len(b_id)), indexing="ij")
        J, K = J.ravel(), K.ravel()
        rd = R.ravel()
        unreach = ~np.isfinite(rd)
        rd_f = np.where(unreach, np.nan, rd)
        lab_ok = true_arc[i] >= 0 and true_arc[i + 1] >= 0
        out.append(pd.DataFrame({
            "pt_from": i, "pt_to": i + 1,
            "arc_from": a_id[J], "arc_to": b_id[K],
            "gc_dist_m": gc, "route_dist_m": rd_f,
            "abs_diff_m": np.abs(rd_f - gc),
            "route_gc_ratio": rd_f / max(gc, 1.0),
            "dt_s": dt,
            "route_speed_kmh": rd_f / max(dt, 1e-3) * 3.6,
            "unreachable": unreach.astype(int),
            "same_arc": (a_id[J] == b_id[K]).astype(int),
            "u_turn": (a_rev[J] == b_id[K]).astype(int),
            "bearing_change_deg": _angle_diff(a_brg[J], b_brg[K]),
            "label": (((a_id[J] == true_arc[i]) & (b_id[K] == true_arc[i + 1])).astype(int)
                      if lab_ok else -1),
        }))
    df = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=TRANSITION_FEATURES + ["label"])
    df.insert(0, "trip", trip.trip_id)
    return df


# ----------------------------------------------------------------------------
# per-trip driver + quality stats
# ----------------------------------------------------------------------------
def build_trip_features(trip: Trip, cfg: FeatureConfig | None = None):
    cfg = cfg or FeatureConfig()
    em = generate_candidates(trip, cfg)
    tr = transition_features(trip, em, cfg)
    pts = trip.points
    labelled = pts.true_arc.values >= 0
    has_true = em[em.label == 1].pt.unique()
    stats = {
        "n_points": len(pts),
        "n_points_labelled": int(labelled.sum()),
        "pct_points_with_candidates": em.pt.nunique() / max(len(pts), 1) * 100,
        # candidate recall: of the labelled points, how often is the true arc a candidate?
        "candidate_recall_pct": len(has_true) / max(labelled.sum(), 1) * 100,
        "mean_candidates": em.groupby("pt").size().mean(),
        "n_emission_rows": len(em),
        "n_transition_rows": len(tr),
        "pct_transitions_unreachable": tr.unreachable.mean() * 100 if len(tr) else np.nan,
        # does the true transition have a finite route distance?
        "true_transition_reachable_pct": (1 - tr[tr.label == 1].unreachable.mean()) * 100
        if (tr.label == 1).any() else np.nan,
    }
    return em, tr, stats


def main(processed_dir: str, out_dir: str, cfg: FeatureConfig | None = None):
    cfg = cfg or FeatureConfig()
    rows = []
    for part, trip_id in list_trips(processed_dir):
        trip = load_trip(processed_dir, part, trip_id)
        em, tr, s = build_trip_features(trip, cfg)
        d = os.path.join(out_dir, part)
        os.makedirs(d, exist_ok=True)
        _write(em, os.path.join(d, f"{trip_id}_emission"))
        _write(tr, os.path.join(d, f"{trip_id}_transition"))
        rows.append({"part": part, "trip": trip_id, **s})
        print(f"{part}/{trip_id}: {s['n_emission_rows']} emission, {s['n_transition_rows']} transition rows, "
              f"recall {s['candidate_recall_pct']:.1f}%", flush=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out_dir, "feature_summary.csv"), index=False)
    with open(os.path.join(out_dir, "feature_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=1)
    for part, g in summ.groupby("part"):
        print(f"\n[{part}] candidate recall {g.candidate_recall_pct.mean():.1f}% (min {g.candidate_recall_pct.min():.1f}%), "
              f"true transition reachable {g.true_transition_reachable_pct.mean():.1f}%, "
              f"{g.n_emission_rows.sum():,} emission / {g.n_transition_rows.sum():,} transition rows")


# ----------------------------------------------------------------------------
# preparation: load a partition as model-ready X, y, groups
# ----------------------------------------------------------------------------
def load_dataset(features_dir: str, part: str, kind: str = "emission", labelled_only: bool = True):
    """Return (X, y, groups, meta) for one partition.

    kind     'emission' or 'transition'
    groups   trip id per row: use GroupKFold so no trip is split across folds
             (Section 7 leakage rule).
    Scaling / imputation is NOT done here on purpose: put it inside the model
    pipeline (e.g. sklearn Pipeline) so it is fitted on training folds only."""
    d = os.path.join(features_dir, part)
    files = sorted({f.split(".")[0] for f in os.listdir(d) if f"_{kind}." in f})
    df = pd.concat([_read(os.path.join(d, f)) for f in files], ignore_index=True)
    df["trip"] = df.trip.astype(str).str.zfill(8)
    if labelled_only:
        df = df[df.label >= 0]
    cols = EMISSION_FEATURES if kind == "emission" else TRANSITION_FEATURES
    meta_cols = [c for c in df.columns if c not in cols and c != "label"]
    return df[cols], df.label.astype(int), df.trip, df[meta_cols]


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
