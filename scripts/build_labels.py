"""Align every train / student_test route to its GPS fixes; save derived labels.

python scripts/build_labels.py -> outputs/labels/<partition>/<trip>.npz, outputs/labels/alignment_report.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.align import align_route  # noqa: E402
from pathsnap.io import PARTITIONS, list_trips, load_trip  # noqa: E402
from pathsnap.labels import load_route  # noqa: E402
from pathsnap.roadnet import RoadNet  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "labels")
rows = []
for part in PARTITIONS:
    os.makedirs(os.path.join(OUT, part), exist_ok=True)
    for tid in list_trips(part):
        t0 = time.time()
        tr = load_trip(part, tid)
        rn = RoadNet(tr)
        xy = rn.project_track(tr.track)
        r = align_route(rn, xy, tr.track[:, 2], load_route(part, tid))
        np.savez_compressed(os.path.join(OUT, part, f"{tid}.npz"), **r)
        K = len(r["route_canon"])
        used = len(np.unique(r["route_pos"]))
        rows.append(dict(partition=part, trip=tid, n_obs=len(xy), route_len=K,
                         route_pos_used=used, route_cover=used / K,
                         d_med=np.median(r["dist"]), d_p95=np.quantile(r["dist"], .95),
                         d_p99=np.quantile(r["dist"], .99), d_max=r["dist"].max(),
                         far=r["far"].mean(), boundary=r["boundary"].mean(),
                         after_gap=r["after_gap"].mean(), chain_break=r["chain_break"].mean(),
                         ambiguous=r["ambiguous"].mean(), sec=time.time() - t0))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "alignment_report.csv"), index=False)
for part, g in df.groupby("partition"):
    w = g.n_obs / g.n_obs.sum()
    print(part, "obs", g.n_obs.sum(), {c: round(float((g[c] * w).sum()), 4) for c in
                                       ["far", "boundary", "after_gap", "chain_break", "ambiguous", "route_cover"]})
