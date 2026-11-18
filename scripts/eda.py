"""Dataset inventory and quality checks (train + student_test only).

python scripts/eda.py  ->  outputs/eda/inventory.csv, outputs/eda/summary.txt
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathsnap.io import PARTITIONS, list_trips, load_trip, trip_dir  # noqa: E402
from pathsnap.labels import load_route  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "eda")
os.makedirs(OUT, exist_ok=True)

rows = []
all_dt = []
for part in PARTITIONS:
    for tid in list_trips(part):
        tr = load_trip(part, tid)
        route = load_route(part, tid)
        t = tr.track[:, 2]
        dt = np.diff(t)
        all_dt.append(dt)
        lon, lat = tr.track[:, 0], tr.track[:, 1]
        A, N = len(tr.arcs), len(tr.nodes)
        fr, to = tr.arcs[:, 0], tr.arcs[:, 1]
        # route chain check: to-node of arc k == from-node of arc k+1
        rf, rt = fr[route], to[route]
        chained = (rt[:-1] == rf[1:])
        g = sp.coo_matrix((np.ones(A), (fr, to)), shape=(N, N)).tocsr()
        ncc_w, lab_w = connected_components(g, directed=True, connection="weak")
        ncc_s, lab_s = connected_components(g, directed=True, connection="strong")
        deg = np.bincount(fr, minlength=N) + np.bincount(to, minlength=N)
        # reverse arc availability
        key = fr * N + to
        rkey = to * N + fr
        has_rev = np.isin(rkey, key)
        # duplicates
        dup_arcs = A - len(np.unique(key))
        # step distance (rough, equirectangular)
        lat0 = np.deg2rad(lat.mean())
        x = np.deg2rad(lon) * 6371000 * np.cos(lat0)
        y = np.deg2rad(lat) * 6371000
        step = np.hypot(np.diff(x), np.diff(y))
        rows.append(dict(
            partition=part, trip=tid, n_obs=len(t), n_nodes=N, n_arcs=A, route_len=len(route),
            route_unique=len(np.unique(route)), route_chain_breaks=int((~chained).sum()),
            route_selfloop_or_repeat_consec=int((np.diff(route) == 0).sum()),
            t0=t[0], duration_s=t[-1] - t[0], dt_min=dt.min(), dt_median=np.median(dt), dt_max=dt.max(),
            frac_dt_1=float((dt == 1).mean()), n_gaps_gt5=int((dt > 5).sum()), n_gaps_gt30=int((dt > 30).sum()),
            nonmonotone_t=int((dt <= 0).sum()), nan_rows=int(np.isnan(tr.track).any(1).sum()),
            dup_positions=int(((np.diff(lon) == 0) & (np.diff(lat) == 0)).sum()),
            max_step_m=step.max(), median_step_m=np.median(step), max_speed_mps=(step / np.maximum(dt, 1e-9)).max(),
            lon_min=lon.min(), lon_max=lon.max(), lat_min=lat.min(), lat_max=lat.max(),
            node_lon_min=tr.nodes[:, 0].min(), node_lon_max=tr.nodes[:, 0].max(),
            node_lat_min=tr.nodes[:, 1].min(), node_lat_max=tr.nodes[:, 1].max(),
            weak_cc=ncc_w, largest_weak_cc_frac=np.bincount(lab_w).max() / N, strong_cc=ncc_s,
            isolated_nodes=int((deg == 0).sum()), selfloop_arcs=int((fr == to).sum()), dup_arcs=dup_arcs,
            frac_arcs_with_reverse=float(has_rev.mean()),
            route_arc_out_of_range=int(((route < 0) | (route >= A)).sum()),
            size_track_kb=os.path.getsize(os.path.join(trip_dir(part, tid), f"{tid}.track")) / 1024,
            size_arcs_kb=os.path.getsize(os.path.join(trip_dir(part, tid), f"{tid}.arcs")) / 1024,
        ))
        print(part, tid, len(t), N, A, len(route), flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "inventory.csv"), index=False)
dt_all = np.concatenate(all_dt)
with open(os.path.join(OUT, "summary.txt"), "w") as f:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    f.write(df.groupby("partition").agg(trips=("trip", "count"), obs=("n_obs", "sum"),
                                         obs_med=("n_obs", "median"), route_med=("route_len", "median"),
                                         nodes_med=("n_nodes", "median"), arcs_med=("n_arcs", "median")).to_string())
    f.write("\n\nDescribe:\n" + df.describe().T.to_string())
    qs = [0.5, 0.9, 0.95, 0.97, 0.99, 0.999]
    f.write("\n\ndt quantiles (all trips): " + str({q: float(np.quantile(dt_all, q)) for q in qs}))
    vals, cnts = np.unique(np.round(dt_all, 3), return_counts=True)
    order = np.argsort(-cnts)[:15]
    f.write("\nmost common dt: " + str([(float(vals[i]), int(cnts[i])) for i in order]))
    f.write(f"\nfrac dt==1: {np.mean(dt_all == 1):.4f}\n")
print(open(os.path.join(OUT, "summary.txt")).read())
