"""Candidate recall vs radius / max_k on TRAIN trips (label-reliable trips only).

python scripts/candidate_recall.py -> outputs/candidates/recall_grid.csv
Denominator: all fixes of label-reliable training trips (and separately the
non-ambiguous subset). A miss = derived label not in the candidate set.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.candidates import generate, recall_table  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.roadnet import RoadNet  # noqa: E402
from pathsnap.splits import BAD_LABEL_TRIPS  # noqa: E402

OUT = os.path.join(ROOT, "outputs", "candidates")
os.makedirs(OUT, exist_ok=True)
grid = [(r, k) for r in (25, 35, 50, 75, 100) for k in (4, 8, 12, 16, 24, 32)]
acc = {g: [0, 0, 0, 0, 0] for g in grid}
for tid in list_trips("train"):
    if tid in BAD_LABEL_TRIPS:
        continue
    tr = load_trip("train", tid)
    rn = RoadNet(tr)
    xy = rn.project_track(tr.track)
    L = np.load(os.path.join(ROOT, "outputs", "labels", "train", f"{tid}.npz"))
    lab, amb = L["label"], L["ambiguous"]
    for (r, k) in grid:
        c = generate(rn, xy, r, k)
        hit, rank, n = recall_table(c, lab)
        a = acc[(r, k)]
        a[0] += hit.sum(); a[1] += len(hit); a[2] += hit[~amb].sum(); a[3] += (~amb).sum(); a[4] += n.sum()
    print(tid, flush=True)
rows = [dict(radius=r, max_k=k, recall_all=a[0] / a[1], recall_clean=a[2] / a[3], mean_cands=a[4] / a[1],
             n_fixes=a[1], n_clean=a[3]) for (r, k), a in acc.items()]
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "recall_grid.csv"), index=False)
print(df.pivot(index="radius", columns="max_k", values="recall_all").round(4))
print(df.pivot(index="radius", columns="max_k", values="mean_cands").round(1))
