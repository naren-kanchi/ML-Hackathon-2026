"""Observation-level log loss on the common evaluation set (team-labelled points,
team labels): -log p(true arc), p clipped at 1e-6 (a candidate miss scores 13.8),
plus the miss rate. Models whose probabilities are saved: LightGBM, IMM+2EKF
(team data), GNN (1 Hz, read off at raw_idx). HGB/LogReg probabilities were not
saved by scripts/viterbi_others.py, so they are not included here.

python scripts/common_logloss.py -> experiments/results/common_eval/log_loss.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.context import prepare  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.roadnet import RoadNet  # noqa: E402
from pathsnap.splits import load_folds  # noqa: E402
from src.data.preprocess import _read  # noqa: E402
from src.data.preprocess import load_trip as team_load  # noqa: E402

PRED = os.path.join(ROOT, "outputs", "preds")
CV = sorted(sum(load_folds()["folds"], []))
ST = list_trips("student_test")


def trip(model, part, tid):
    k = "oof" if part == "train" else "st"
    raw = load_trip(part, tid)
    rn = RoadNet(raw)
    pts = team_load(os.path.join(ROOT, "data", "processed"), part, tid).points
    lab = pts.true_arc.values.astype(int)
    L = lab >= 0
    labc = np.where(L, rn.canon[np.maximum(lab, 0)], -1)
    p_true = np.zeros(len(pts))
    if model == "gnn_mp":
        ctx = prepare(raw)
        c = ctx.cands
        temps = json.load(open(os.path.join(ROOT, "src", "models", "road_gnn_rlcd", "results", "m2_mp", "temperature.json")))
        z = np.load(os.path.join(PRED, "m2_mp", f"{k}_{tid}.npz"))["logits"][2] / temps["2"]["T"]
        p = np.zeros(len(z))
        for f in range(len(c.ptr) - 1):
            s, e = c.ptr[f], c.ptr[f + 1]
            if e > s:
                q = np.exp(z[s:e] - z[s:e].max()); p[s:e] = q / q.sum()
        idx = pts.raw_idx.values.astype(int)
        for j, i in enumerate(idx):
            s, e = c.ptr[i], c.ptr[i + 1]
            m = c.arc[s:e] == labc[j]
            p_true[j] = p[s:e][m].sum() if L[j] else 0
    else:
        em = _read(os.path.join(ROOT, "data", "features", part, f"{tid}_emission"))
        em = em.sort_values(["pt", "dist_m"], kind="stable").reset_index(drop=True)
        if model == "lgbm_ranker":
            t = pd.read_parquet(os.path.join(PRED, "lgbm_ranker", f"{k}_{tid}.parquet"))
            p = em[["pt", "arc_id"]].merge(t, on=["pt", "arc_id"], how="left").proba.fillna(0).values
        else:
            p = np.load(os.path.join(PRED, f"imm_teamdata_{model}", f"{k}_{tid}.npy"))
        hit = rn.canon[em.arc_id.values] == labc[em.pt.values]
        np.add.at(p_true, em.pt.values[hit], p[hit])
    ll = -np.log(np.clip(p_true[L], 1e-6, 1))
    return dict(model=model, part=part, trip=tid, n=int(L.sum()), ll_sum=float(ll.sum()),
                miss=int((p_true[L] == 0).sum()), ll_cov_sum=float(ll[p_true[L] > 0].sum()))


if __name__ == "__main__":
    jobs = [(m, part, t) for m in ("lgbm_ranker", "team_plus_imm", "gnn_mp")
            for part, ts in (("train", CV), ("student_test", ST)) for t in ts]
    df = pd.DataFrame(Parallel(n_jobs=12)(delayed(trip)(*j) for j in jobs))
    g = df.groupby(["model", "part"]).agg(n=("n", "sum"), ll=("ll_sum", "sum"), miss=("miss", "sum"),
                                          llc=("ll_cov_sum", "sum")).reset_index()
    g["log_loss"] = g.ll / g.n
    g["miss_rate"] = g.miss / g.n
    g["log_loss_covered"] = g.llc / (g.n - g.miss)
    g = g[["model", "part", "n", "log_loss", "log_loss_covered", "miss_rate"]]
    g.to_csv(os.path.join(ROOT, "experiments", "results", "common_eval", "log_loss.csv"), index=False)
    print(g.round(4).to_string(index=False))
