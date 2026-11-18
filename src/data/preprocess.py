"""
PathSnap - Data cleaning and preprocessing.

Turns one raw trip (.track/.nodes/.arcs/.route) into a clean, projected,
labelled trip object that the feature and model code can use directly.

Steps (each one comes from a finding in reports/eda and reports/geo):
  1. Read raw files                       (lon, lat order confirmed in EDA)
  2. Project to metres                    (per-trip UTM; Section 5)
  3. Clean the road network               (duplicate arcs, self-loops)
  4. Split the GPS trace at long gaps     (23 trips have gaps > 60 s)
  5. Collapse stationary (parked) runs    (51 trips; trips 33/54 parked off-road)
  6. Drop isolated position spikes        (implied speeds up to 648 km/h)
  7. Thin points closer than min_dist_m   (consecutive 1 Hz points are redundant)
  8. Align each kept point to the reference route (label = true arc_id)

Every threshold lives in PreprocessConfig so it can be tuned later
(Bayesian optimisation) using TRAIN trips only.

Usage (from the repo root):
    python -m src.data.preprocess Dataset data/processed
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

try:
    import pyproj
except ImportError:  # fallback keeps the code runnable without pyproj
    pyproj = None

R_EARTH = 6371008.8


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    gap_split_s: float = 120.0       # start a new segment after a gap longer than this
    stat_window_s: float = 60.0      # stationary if all points in +/- window/2 ...
    stat_box_m: float = 30.0         # ... fit in a box this size
    spike_speed_kmh: float = 250.0   # point is a spike if in AND out speed exceed this
    min_dist_m: float = 10.0         # keep a point only if >= this far from last kept point
    min_segment_points: int = 3      # drop segments shorter than this after cleaning
    label_max_dist_m: float = 50.0   # aligned point farther than this from route -> label -1


# ----------------------------------------------------------------------------
# reading & projection
# ----------------------------------------------------------------------------
def read_raw_trip(trip_dir: str, trip: str):
    f = lambda e: os.path.join(trip_dir, f"{trip}.{e}")
    track = np.loadtxt(f("track"), ndmin=2)                       # lon, lat, t
    nodes = np.loadtxt(f("nodes"), ndmin=2)                       # lon, lat
    arcs = np.loadtxt(f("arcs"), dtype=np.int64, ndmin=2)         # from, to
    route = (np.loadtxt(f("route"), dtype=np.int64, ndmin=1)
             if os.path.exists(f("route")) and os.path.getsize(f("route")) else np.array([], np.int64))
    return track, nodes, arcs, route


class Projector:
    """lon/lat -> metres. Each trip gets its own UTM zone (pyproj); without pyproj,
    a local tangent plane centred on the trip (worst error vs haversine 0.57%)."""

    def __init__(self, lon0: float, lat0: float):
        self.lon0, self.lat0 = float(lon0), float(lat0)
        if pyproj is not None:
            zone = int((lon0 + 180) // 6) + 1
            self.epsg = (32600 if lat0 >= 0 else 32700) + zone
            self._t = pyproj.Transformer.from_crs(4326, self.epsg, always_xy=True)
            self.name = f"UTM zone {zone}{'N' if lat0 >= 0 else 'S'} (EPSG:{self.epsg})"
        else:
            self.epsg, self._t = None, None
            self.name = "local tangent plane"

    def __call__(self, lon, lat):
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        if self._t is not None:
            return np.column_stack(self._t.transform(lon, lat))
        x = R_EARTH * np.radians(lon - self.lon0) * np.cos(np.radians(self.lat0))
        y = R_EARTH * np.radians(lat - self.lat0)
        return np.column_stack([x, y])


# ----------------------------------------------------------------------------
# road network
# ----------------------------------------------------------------------------
def clean_network(nodes_xy: np.ndarray, arcs: np.ndarray):
    """Remove self-loops and duplicate arcs.

    Arc IDs are kept as the original row index (the .route file refers to them).
    Returns a DataFrame indexed by arc_id plus a mapping {dropped_id: kept_id}
    so route arc IDs pointing at a removed duplicate can be redirected."""
    df = pd.DataFrame({"arc_id": np.arange(len(arcs)), "u": arcs[:, 0], "v": arcs[:, 1]})
    canon = {}
    first = df.groupby(["u", "v"]).arc_id.transform("min")
    for a, k in zip(df.arc_id[first != df.arc_id], first[first != df.arc_id]):
        canon[int(a)] = int(k)
    df = df[(first == df.arc_id) & (df.u != df.v)].copy()

    A, B = nodes_xy[df.u.values], nodes_xy[df.v.values]
    d = B - A
    df["length_m"] = np.hypot(d[:, 0], d[:, 1])
    df["bearing_deg"] = np.degrees(np.arctan2(d[:, 0], d[:, 1])) % 360     # 0 = north
    pairs = set(zip(df.u.values, df.v.values))
    df["bidirectional"] = [(v, u) in pairs for u, v in zip(df.u.values, df.v.values)]
    rev = {(u, v): a for a, u, v in zip(df.arc_id.values, df.u.values, df.v.values)}
    df["reverse_arc_id"] = [rev.get((v, u), -1) for u, v in zip(df.u.values, df.v.values)]
    outdeg = np.bincount(df.u.values, minlength=len(nodes_xy))
    indeg = np.bincount(df.v.values, minlength=len(nodes_xy))
    df["to_out_degree"] = outdeg[df.v.values]
    df["from_in_degree"] = indeg[df.u.values]
    df["dead_end"] = df.to_out_degree.eq(0) | (df.to_out_degree.eq(1) & df.bidirectional)
    return df.set_index("arc_id"), canon


# ----------------------------------------------------------------------------
# GPS cleaning
# ----------------------------------------------------------------------------
def stationary_mask(xy: np.ndarray, t: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    s = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}, index=pd.to_datetime(t, unit="s"))
    w = f"{int(cfg.stat_window_s)}s"
    span = lambda c: (s[c].rolling(w, center=True).max() - s[c].rolling(w, center=True).min()).values
    return (span("x") < cfg.stat_box_m) & (span("y") < cfg.stat_box_m)


def collapse_stationary(df: pd.DataFrame, stat: np.ndarray) -> pd.DataFrame:
    """Replace each run of stationary points by one point at the run's median
    position (time = first timestamp, 'stop_s' = how long the vehicle stood)."""
    run = np.cumsum(np.r_[True, stat[1:] != stat[:-1]])
    df = df.assign(_run=run, _stat=stat)
    out = []
    for _, g in df.groupby("_run", sort=True):
        if g._stat.iloc[0] and len(g) > 1:
            r = g.iloc[[0]].copy()
            r["x"], r["y"] = g.x.median(), g.y.median()
            r["lon"], r["lat"] = g.lon.median(), g.lat.median()
            r["stop_s"] = g.t.iloc[-1] - g.t.iloc[0]
            r["n_raw"] = len(g)
            out.append(r)
        else:
            out.append(g.assign(stop_s=0.0, n_raw=1))
    return pd.concat(out).drop(columns=["_run", "_stat"])


def drop_spikes(df: pd.DataFrame, cfg: PreprocessConfig) -> tuple[pd.DataFrame, int]:
    """Remove single points whose implied speed is impossible both into and out
    of the point (a jump away and straight back)."""
    if len(df) < 3:
        return df, 0
    xy, t = df[["x", "y"]].values, df.t.values
    v_in = np.r_[0, np.hypot(*np.diff(xy, axis=0).T) / np.maximum(np.diff(t), 1e-3)] * 3.6
    v_out = np.r_[v_in[1:], 0]
    skip = np.r_[0, np.hypot(*(xy[2:] - xy[:-2]).T) / np.maximum(t[2:] - t[:-2], 1e-3) * 3.6, 0]
    spike = (v_in > cfg.spike_speed_kmh) & (v_out > cfg.spike_speed_kmh) & (skip < cfg.spike_speed_kmh)
    return df[~spike], int(spike.sum())


def thin(df: pd.DataFrame, min_dist: float) -> pd.DataFrame:
    if min_dist <= 0 or len(df) < 2:
        return df
    xy = df[["x", "y"]].values
    keep, last = [0], xy[0]
    for i in range(1, len(xy) - 1):
        if np.hypot(*(xy[i] - last)) >= min_dist:
            keep.append(i); last = xy[i]
    keep.append(len(xy) - 1)                                   # always keep the end point
    return df.iloc[keep]


# ----------------------------------------------------------------------------
# reference-route alignment (labels)
# ----------------------------------------------------------------------------
def point_segment_dist(P, A, B):
    """Distances from every point in P (n,2) to every segment A-B (m,2) -> (n,m),
    plus the projection fraction t in [0,1] along each segment."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    t = np.clip(((P[:, None, :] - A) * AB).sum(2) / L2, 0, 1)
    proj = A + t[..., None] * AB
    return np.sqrt(((P[:, None, :] - proj) ** 2).sum(2)), t


def align_to_route(xy: np.ndarray, route_xy_a: np.ndarray, route_xy_b: np.ndarray, chunk: int = 512):
    """Monotonic alignment of GPS points to route positions.

    The route is an ordered list of arcs, so point i+1 must map to the same or a
    later route position than point i. Minimises total point-to-arc distance with
    dynamic programming (cost[i,j] = d[i,j] + min_{k<=j} cost[i-1,k]).
    Handles loops/revisits correctly, which nearest-arc snapping does not."""
    n, m = len(xy), len(route_xy_a)
    back = np.empty((n, m), np.int32)
    prev = None
    dist_rows = np.empty((n, m), np.float32)
    for s in range(0, n, chunk):
        d, _ = point_segment_dist(xy[s:s + chunk], route_xy_a, route_xy_b)
        dist_rows[s:s + chunk] = d
    for i in range(n):
        if prev is None:
            cost = dist_rows[i].astype(np.float64)
            back[i] = np.arange(m)
        else:
            # running minimum of the previous row and where it came from
            idx = np.arange(m)
            run_min = np.minimum.accumulate(prev)
            arg = np.maximum.accumulate(np.where(prev <= run_min, idx, 0))
            cost = dist_rows[i] + prev[arg]
            back[i] = arg
        prev = cost
    pos = np.empty(n, np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    return pos, dist_rows[np.arange(n), pos]


# ----------------------------------------------------------------------------
# main entry: one trip
# ----------------------------------------------------------------------------
@dataclass
class Trip:
    trip_id: str
    part: str
    projection: str
    points: pd.DataFrame          # cleaned, thinned GPS points (one row per kept point)
    arcs: pd.DataFrame            # cleaned network, index = arc_id
    nodes_xy: np.ndarray          # (n_nodes, 2) metres
    route: np.ndarray             # reference route arc_ids (canonicalised), may be empty
    stats: dict = field(default_factory=dict)


def preprocess_trip(trip_dir: str, trip: str, part: str, cfg: PreprocessConfig | None = None) -> Trip:
    cfg = cfg or PreprocessConfig()
    track, nodes, arcs_raw, route_raw = read_raw_trip(trip_dir, trip)
    prj = Projector(np.median(track[:, 0]), np.median(track[:, 1]))
    nodes_xy = prj(nodes[:, 0], nodes[:, 1])
    arcs, canon = clean_network(nodes_xy, arcs_raw)
    route = np.array([canon.get(int(a), int(a)) for a in route_raw], np.int64)
    route = route[np.isin(route, arcs.index.values)]           # drops removed self-loops only

    xy = prj(track[:, 0], track[:, 1])
    raw = pd.DataFrame({"raw_idx": np.arange(len(track)), "lon": track[:, 0], "lat": track[:, 1],
                        "x": xy[:, 0], "y": xy[:, 1], "t": track[:, 2]})
    stats = {"n_raw": len(raw), "n_arcs_raw": len(arcs_raw), "n_arcs_clean": len(arcs),
             "dup_arcs_removed": len(canon), "self_loops_removed": len(arcs_raw) - len(canon) - len(arcs)}

    # 4. split at long gaps
    seg = np.r_[0, np.cumsum(np.diff(raw.t.values) > cfg.gap_split_s)]
    raw["segment"] = seg
    stats["n_segments_gap"] = int(seg.max() + 1)

    # 5-7 per segment
    stat = stationary_mask(xy, raw.t.values, cfg)
    stats["pct_stationary_raw"] = float(stat.mean() * 100)
    pieces, n_spikes = [], 0
    for s, g in raw.groupby("segment"):
        g = collapse_stationary(g, stat[g.index.values])
        g, k = drop_spikes(g, cfg); n_spikes += k
        g = thin(g, cfg.min_dist_m)
        if len(g) >= cfg.min_segment_points:
            pieces.append(g)
    pts = pd.concat(pieces, ignore_index=True)
    pts["segment"] = pts.groupby("segment").ngroup()             # renumber 0..k-1
    stats.update(n_spikes_removed=n_spikes, n_kept=len(pts), n_segments=int(pts.segment.nunique()),
                 pct_kept=len(pts) / len(raw) * 100)

    # per-point kinematics inside each segment
    pts["dt_prev"] = pts.groupby("segment").t.diff()
    d = np.hypot(pts.x.diff(), pts.y.diff())
    pts["dist_prev_m"] = np.where(pts.dt_prev.notna(), d, np.nan)
    pts["speed_prev_mps"] = pts.dist_prev_m / pts.dt_prev.where(pts.dt_prev > 0)
    # heading from previous to next kept point (central difference) inside the segment
    nx_ = pts.groupby("segment").x.shift(-1).fillna(pts.x); px_ = pts.groupby("segment").x.shift(1).fillna(pts.x)
    ny_ = pts.groupby("segment").y.shift(-1).fillna(pts.y); py_ = pts.groupby("segment").y.shift(1).fillna(pts.y)
    hx, hy = nx_ - px_, ny_ - py_
    pts["heading_deg"] = np.where(np.hypot(hx, hy) > 1.0, np.degrees(np.arctan2(hx, hy)) % 360, np.nan)

    # 8. labels from the reference route
    pts["true_arc"] = -1
    pts["route_pos"] = -1
    pts["dist_to_true_m"] = np.nan
    if len(route):
        ra = arcs.loc[route, ["u", "v"]].values
        pos, dist = align_to_route(pts[["x", "y"]].values, nodes_xy[ra[:, 0]], nodes_xy[ra[:, 1]])
        ok = dist <= cfg.label_max_dist_m
        pts["route_pos"] = np.where(ok, pos, -1)
        pts["true_arc"] = np.where(ok, route[pos], -1)
        pts["dist_to_true_m"] = dist
        stats["pct_labelled"] = float(ok.mean() * 100)
    stats["projection"] = prj.name
    return Trip(trip, part, prj.name, pts, arcs, nodes_xy, route, stats)


# ----------------------------------------------------------------------------
# saving / loading
# ----------------------------------------------------------------------------
def _write(df: pd.DataFrame, path_no_ext: str, index=False):
    try:
        df.to_parquet(path_no_ext + ".parquet", index=index)
    except (ImportError, ValueError):
        df.to_csv(path_no_ext + ".csv.gz", index=index)


def _read(path_no_ext: str, index_col=None):
    if os.path.exists(path_no_ext + ".parquet"):
        df = pd.read_parquet(path_no_ext + ".parquet")
        return df.set_index(index_col) if index_col and index_col in df.columns else df
    return pd.read_csv(path_no_ext + ".csv.gz", index_col=index_col)


def save_trip(trip: Trip, out_dir: str):
    d = os.path.join(out_dir, trip.part, trip.trip_id)
    os.makedirs(d, exist_ok=True)
    _write(trip.points, os.path.join(d, "points"))
    _write(trip.arcs.reset_index(), os.path.join(d, "arcs"))
    np.save(os.path.join(d, "nodes_xy.npy"), trip.nodes_xy)
    np.save(os.path.join(d, "route.npy"), trip.route)
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump({"trip_id": trip.trip_id, "part": trip.part, "projection": trip.projection,
                   "stats": trip.stats}, fh, indent=1)


def load_trip(processed_dir: str, part: str, trip_id: str) -> Trip:
    d = os.path.join(processed_dir, part, trip_id)
    with open(os.path.join(d, "meta.json")) as fh:
        meta = json.load(fh)
    return Trip(trip_id, part, meta["projection"], _read(os.path.join(d, "points")),
                _read(os.path.join(d, "arcs"), index_col="arc_id"),
                np.load(os.path.join(d, "nodes_xy.npy")), np.load(os.path.join(d, "route.npy")),
                meta["stats"])


def list_trips(root: str):
    for part in sorted(p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))):
        for trip in sorted(os.listdir(os.path.join(root, part))):
            if os.path.isdir(os.path.join(root, part, trip)):
                yield part, trip


def main(raw_root: str, out_dir: str, cfg: PreprocessConfig | None = None):
    cfg = cfg or PreprocessConfig()
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for part, trip in list_trips(raw_root):
        tr = preprocess_trip(os.path.join(raw_root, part, trip), trip, part, cfg)
        save_trip(tr, out_dir)
        rows.append({"part": part, "trip": trip, **tr.stats})
        s = tr.stats
        print(f"{part}/{trip}: {s['n_raw']} -> {s['n_kept']} pts, {s['n_segments']} seg, "
              f"{s.get('pct_labelled', float('nan')):.1f}% labelled", flush=True)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out_dir, "preprocess_summary.csv"), index=False)
    with open(os.path.join(out_dir, "preprocess_config.json"), "w") as fh:
        json.dump(asdict(cfg), fh, indent=1)
    print(f"\nkept {summ.n_kept.sum()} of {summ.n_raw.sum()} points "
          f"({summ.n_kept.sum() / summ.n_raw.sum() * 100:.1f}%), "
          f"labelled {summ.pct_labelled.mean():.1f}% on average")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
