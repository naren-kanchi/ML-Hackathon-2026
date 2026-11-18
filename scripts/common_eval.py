"""Common-set comparison of ALL models: same points, same labels, same decoder.

Why: each model's own evaluation uses different points/labels (raw 1 Hz fixes +
shared derived labels; or the team's cleaned/thinned points + team labels), so
their headline numbers are not comparable. Here every model is scored on ONE set:

  points  = the team-preprocessed points that carry a team label (true_arc >= 0),
            i.e. data/processed/<part>/<trip>/points (raw_idx -> raw 1 Hz fix)
  labels  = team true_arc, mapped to canonical arc ids (duplicates merged)
  decoder = pathsnap/hmm.py with default HMMParams for every model

Models on raw 1 Hz fixes (IMM 1 Hz, GNN nomp/mp, HGB, logistic) are decoded on
their full fix sequence and read off at each team point's raw_idx. Models on the
team points (LightGBM ranker, team-features MLP, team+IMM MLP) are decoded on
those points directly. CV = out-of-fold predictions of the 78 frozen-fold trips;
student-test = final refits.

Metrics per (model, split, mode):
  top1        correct / team-labelled points (misses and no-candidate points = wrong)
  macro_f1    per-trip arc-class macro F1 (labelled points), mean over trips
  fold_top1   mean and sd of the per-fold point-weighted Top-1 (CV only)
  trips_won   number of trips where this model has the best Top-1 (ties shared)

python scripts/common_eval.py -> experiments/results/common_eval/
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.context import prepare  # noqa: E402
from pathsnap.hmm import classifier_only, viterbi  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from pathsnap.metrics import macro_f1_trip  # noqa: E402
from pathsnap.roadnet import RoadNet  # noqa: E402
from pathsnap.splits import load_folds  # noqa: E402
from src.data.preprocess import load_trip as team_load  # noqa: E402
from src.models.imm_2ekf.teamdata import ctx_from_candidate_lists  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "results", "common_eval")
PRED = os.path.join(ROOT, "outputs", "preds")
PROC, FEAT = os.path.join(ROOT, "data", "processed"), os.path.join(ROOT, "data", "features")
FOLDS = load_folds()["folds"]
FOLD = {t: k for k, f in enumerate(FOLDS) for t in f}
CV, ST = sorted(FOLD), list_trips("student_test")


def team_points(part, tid, rn):
    pts = team_load(PROC, part, tid).points
    lab = pts.true_arc.values.astype(int)
    lab = np.where(lab >= 0, rn.canon[np.maximum(lab, 0)], -1)
    return pts, lab


def softmax_flat(z, ptr, T):
    out = np.zeros(len(z))
    for f in range(len(ptr) - 1):
        s, e = ptr[f], ptr[f + 1]
        if e > s:
            a = np.exp(z[s:e] / T - (z[s:e] / T).max())
            out[s:e] = a / a.sum()
    return out


# ---------------------------------------------------------------- per-model arc sources
def arcs_1hz(model, part, tid):
    """Decoded arcs on all raw fixes for the pathsnap 1 Hz models."""
    ctx = prepare(load_trip(part, tid))
    k = "oof" if part == "train" else "st"
    if model == "imm_1hz":
        p = np.load(os.path.join(PRED, "m1", f"{k}_{tid}.npy"))
    else:
        v = model.split("_")[1]
        R = os.path.join(ROOT, "src", "models", "road_gnn_rlcd", "results", f"m2_{v}")
        temps = json.load(open(os.path.join(R, "temperature.json")))
        p = softmax_flat(np.load(os.path.join(PRED, f"m2_{v}", f"{k}_{tid}.npz"))["logits"][2],
                         ctx.cands.ptr, temps["2"]["T"])
    return classifier_only(ctx, p)["arc"], viterbi(ctx, p)["arc"], ctx.rn


def arcs_midpoint(model, part, tid):
    z = np.load(os.path.join(PRED, f"arcs_{model}", f"{part}_{tid}.npz"))
    return z["clf"], z["hmm"], RoadNet(load_trip(part, tid))


def team_probs(model, part, tid, em):
    k = "oof" if part == "train" else "st"
    if model == "lgbm_ranker":
        t = pd.read_parquet(os.path.join(PRED, "lgbm_ranker", f"{k}_{tid}.parquet"))
        return em[["pt", "arc_id"]].merge(t, on=["pt", "arc_id"], how="left").proba.fillna(0.0).values
    return np.load(os.path.join(PRED, f"imm_teamdata_{model}", f"{k}_{tid}.npy"))


def eval_trip(model, kind, part, tid):
    if kind == "team":
        from src.data.preprocess import _read
        from pathsnap.io import Trip
        raw = load_trip(part, tid)
        rn = RoadNet(raw)
        pts, lab = team_points(part, tid, rn)
        em = _read(os.path.join(FEAT, part, f"{tid}_emission")).sort_values(["pt", "dist_m"], kind="stable")
        em = em.reset_index(drop=True)
        p = team_probs(model, part, tid, em)
        n = len(pts)
        ptr = np.searchsorted(em.pt.values, np.arange(n + 1))
        trip = Trip(tid, np.column_stack([pts.lon.values, pts.lat.values, pts.t.values]), raw.nodes, raw.arcs)
        ctx, pr = ctx_from_candidate_lists(trip, rn, [em.arc_id.values[ptr[i]:ptr[i + 1]] for i in range(n)],
                                           [p[ptr[i]:ptr[i + 1]] for i in range(n)])
        a_clf, a_hmm = classifier_only(ctx, pr)["arc"], viterbi(ctx, pr)["arc"]
    else:
        a_clf, a_hmm, rn = (arcs_1hz if kind == "1hz" else arcs_midpoint)(model, part, tid)
        pts, lab = team_points(part, tid, rn)
        idx = pts.raw_idx.values.astype(int)
        a_clf, a_hmm = a_clf[idx], a_hmm[idx]
    L = lab >= 0
    rows = []
    for mode, a in (("clf", a_clf), ("hmm", a_hmm)):
        ok = (a[L] == lab[L])
        rows.append(dict(model=model, part=part, trip=tid, mode=mode, n=int(L.sum()), correct=int(ok.sum()),
                         top1=float(ok.mean()) if L.any() else np.nan,
                         macro_f1=macro_f1_trip(a[L], lab[L]) if L.any() else np.nan))
    return rows


MODELS = [("imm_1hz", "1hz"), ("gnn_nomp", "1hz"), ("gnn_mp", "1hz"),
          ("histgradientboosting", "mid"), ("logistic_regression", "mid"),
          ("lgbm_ranker", "team"), ("team_only", "team"), ("team_plus_imm", "team")]
LABEL = {"imm_1hz": "IMM+2EKF (raw 1 Hz, shared candidates)", "gnn_nomp": "GNN no message passing (1 Hz)",
         "gnn_mp": "GNN message passing (1 Hz)", "histgradientboosting": "HistGradientBoosting (team)",
         "logistic_regression": "Logistic Regression (team)", "lgbm_ranker": "LightGBM LambdaRank (team)",
         "team_only": "Team features + listwise MLP", "team_plus_imm": "IMM+2EKF (team features + IMM, MLP)"}


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = [(m, k, part, t) for m, k in MODELS for part, trips in (("train", CV), ("student_test", ST))
            for t in trips]
    res = Parallel(n_jobs=12, verbose=2)(delayed(eval_trip)(*j) for j in jobs)
    df = pd.DataFrame([r for o in res for r in o])
    df.to_csv(os.path.join(OUT, "per_trip.csv"), index=False)
    rows = []
    for (m, part, mode), g in df.groupby(["model", "part", "mode"]):
        r = dict(model=m, name=LABEL[m], split="cv_oof" if part == "train" else "student_test", mode=mode,
                 trips=len(g), points=int(g.n.sum()), top1=g.correct.sum() / g.n.sum(),
                 macro_f1_tripmean=g.macro_f1.mean())
        if part == "train":
            f = g.assign(fold=g.trip.map(FOLD)).groupby("fold").apply(
                lambda x: x.correct.sum() / x.n.sum(), include_groups=False)
            r.update(fold_top1_mean=f.mean(), fold_top1_sd=f.std(), fold_top1=[round(v, 4) for v in f])
        rows.append(r)
    s = pd.DataFrame(rows)
    # trips won (best Top-1 per trip, ties shared) within split x mode
    for (part, mode), g in df.groupby(["part", "mode"]):
        piv = g.pivot(index="trip", columns="model", values="top1")
        best = piv.max(axis=1)
        wins = (piv.eq(best, axis=0)).sum()
        sp = "cv_oof" if part == "train" else "student_test"
        for m, w in wins.items():
            s.loc[(s.model == m) & (s.split == sp) & (s["mode"] == mode), "trips_best"] = int(w)
    s = s.sort_values(["split", "mode", "top1"], ascending=[True, True, False])
    s.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print(s.drop(columns=["model"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
