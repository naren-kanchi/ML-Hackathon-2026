"""Fit tracker noise parameters on TRAINING trips only (no labels needed).

Objective: mean one-step-ahead predictive log-likelihood of GPS fixes under the
turning EKF (the CT model shared by variants B and C). The IMM then REUSES
sigma_z, sa, spsi_ct, sw and adds only its own straight-model heading noise and
switch rate (fixed, documented). Result -> src/models/imm_2ekf/configs/tracker_1hz.json
"""
import itertools
import json
import os
import sys

import numpy as np
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pathsnap.io import load_trip  # noqa: E402
from pathsnap.roadnet import RoadNet  # noqa: E402
from pathsnap.splits import SEED, load_folds  # noqa: E402
from src.models.imm_2ekf.trackers import TrackerParams, run_tracker  # noqa: E402

folds = load_folds()
trips = sorted(sum(folds["folds"], []))
rng = np.random.default_rng(SEED)
sub = sorted(rng.choice(trips, 24, replace=False).tolist())
data = []
for tid in sub:
    tr = load_trip("train", tid)
    rn = RoadNet(tr)
    xy, t = rn.project_track(tr.track), tr.track[:, 2]
    data.append((xy[:3000], t[:3000]))


def score(sz, sa, sw, kind="ekf"):
    p = TrackerParams(sigma_z=sz, sa=sa, sw=sw)
    tot, n = 0.0, 0
    for xy, t in data:
        o = run_tracker(xy, t, kind, p)
        tot += o["loglik"][o["ready"]].sum()
        n += o["ready"].sum()
    return dict(sigma_z=sz, sa=sa, sw=sw, kind=kind, mean_loglik=tot / n)


grid = list(itertools.product([3.0, 4.0, 5.0, 6.0], [1.0, 1.5, 2.0, 3.0], [0.01, 0.02, 0.05, 0.15]))
res = Parallel(n_jobs=12)(delayed(score)(*g) for g in grid)
res.sort(key=lambda r: -r["mean_loglik"])
for r in res[:8]:
    print(r)
best = res[0]
imm = score(best["sigma_z"], best["sa"], best["sw"], "imm")
print("IMM with shared params:", imm)
cfg = dict(TrackerParams().__dict__)
cfg.update(sigma_z=best["sigma_z"], sa=best["sa"], sw=best["sw"])
json.dump(dict(params=cfg, fit_trips=sub, grid_results=res, imm_check=imm,
               note="sigma_z, sa, sw fitted (EKF predictive loglik, train only); spsi_cv, spsi_ct, p_switch fixed"),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "tracker_1hz.json"), "w"), indent=1)
