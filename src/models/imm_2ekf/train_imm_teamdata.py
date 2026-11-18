"""IMM + 2 EKF candidate scorer on the team's preprocessed data, 5-fold CV.

Run from the repo root, after the team preprocessing + feature build:
    python -m src.models.imm_2ekf.train_imm_teamdata [--processed data/processed] [--features data/features]

Pipeline
 1. Fit IMM noise (sigma_z, sa, sw) on TRAINING trips only: maximise the one-step
    predictive log-likelihood of the turning EKF on the team's cleaned points
    (same procedure as fit_tracker_noise.py on raw 1 Hz data). The IMM reuses them.
 2. Features per candidate row: team emission features (17) + IMM features (12).
 3. Classifier: listwise MLP (pathsnap/listwise.py) = masked softmax over each
    point's candidates, cross-entropy on the derived label.
 4. 5-fold trip-grouped CV with the FROZEN folds (outputs/splits/folds.json);
    out-of-fold predictions -> CV metrics; one temperature fit on OOF.
 5. Refit on all CV trips -> student-test (10 trips) metrics.
 Baseline "team_only": identical classifier/folds/seed on the 17 team features,
 so the IMM's incremental contribution is visible.

Metrics (denominator = team-labelled points, true_arc >= 0):
  obs_top1          top-ranked candidate == label; points whose label is not among
                    candidates (or with no candidates) count as errors
  macro_f1_trip     per-trip arc-class macro F1, mean over trips (no pooling of arc ids)
  obs_log_loss      -log p(label), p clipped at 1e-6 (miss -> 13.8)
  cand_macro_f1     candidate-row binary macro F1 at p >= 0.5 (the HGB README convention)
  cand_log_loss     candidate-row binary log loss
  miss_rate         labelled points whose label is not a candidate
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import f1_score, log_loss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)
from pathsnap.listwise import ListwiseScorer, fit_temperature, pad, unpad  # noqa: E402
from pathsnap.splits import SEED, load_folds  # noqa: E402
from src.data.preprocess import load_trip  # noqa: E402

from src.models.imm_2ekf.teamdata import trip_table  # noqa: E402
from src.models.imm_2ekf.trackers import TrackerParams, run_tracker  # noqa: E402

REINIT_GAP_S = 30.0   # team points: dt p99 = 10 s, only 0.2% of steps > 30 s (train)
RES = os.path.join(HERE, "results", "imm_teamdata")
ART = os.path.join(HERE, "artifacts")
torch.set_num_threads(8)


def fit_noise(processed, trips):
    rng = np.random.default_rng(SEED)
    sub = sorted(rng.choice(trips, min(24, len(trips)), replace=False).tolist())
    data = []
    for t in sub:
        pts = load_trip(processed, "train", t).points
        for s in np.unique(pts.segment.values):
            g = pts[pts.segment == s]
            data.append((g[["x", "y"]].values, g.t.values))

    def score(sz, sa, sw):
        p = TrackerParams(sigma_z=sz, sa=sa, sw=sw)
        tot = n = 0.0
        for xy, t in data:
            o = run_tracker(xy, t, "ekf", p, REINIT_GAP_S)
            tot += o["loglik"][o["ready"]].sum()
            n += o["ready"].sum()
        return dict(sigma_z=sz, sa=sa, sw=sw, mean_loglik=float(tot / n))
    grid = list(itertools.product([3.0, 4.0, 5.0, 6.0, 8.0], [0.5, 1.0, 1.5, 2.0, 3.0], [0.01, 0.02, 0.05, 0.15]))
    res = sorted(Parallel(n_jobs=12)(delayed(score)(*g) for g in grid), key=lambda r: -r["mean_loglik"])
    b = res[0]
    p = TrackerParams(sigma_z=b["sigma_z"], sa=b["sa"], sw=b["sw"])
    return p, dict(params=p.__dict__, fit_trips=sub, reinit_gap_s=REINIT_GAP_S, grid_results=res,
                   note="sigma_z, sa, sw fitted on train (EKF predictive loglik); spsi_cv, spsi_ct, p_switch fixed")


def to_padded(tab, train=False):
    n = tab["n_points"]
    ptr = np.searchsorted(tab["pt"], np.arange(n + 1))
    pos = np.nonzero(tab["label"] == 1)[0]
    y_row = np.full(n, -1)
    y_row[tab["pt"][pos]] = pos
    X, M, y = pad(tab["F"], ptr, y_row)
    lab_ok = tab["true_arc"] >= 0
    if train:
        y = np.where(lab_ok, y, -1)
    return X, M, y, ptr, lab_ok


def trip_metrics(tab, ptr, probs_flat, lab_ok):
    n = tab["n_points"]
    pred = np.full(n, -1)
    p_true = np.zeros(n)
    for i in range(n):
        s, e = ptr[i], ptr[i + 1]
        if e > s:
            pred[i] = tab["arc"][s + int(np.argmax(probs_flat[s:e]))]
    pos = tab["label"] == 1
    p_true[tab["pt"][pos]] = probs_flat[pos]
    ta = tab["true_arc"]
    L = lab_ok
    correct = (pred == ta) & L
    hit = np.zeros(n, bool)
    hit[tab["pt"][pos]] = True
    cls = np.union1d(np.unique(ta[L]), np.unique(pred[L & (pred >= 0)]))
    f1s = []
    for k in cls:
        tp = np.sum((pred[L] == k) & (ta[L] == k)); fp = np.sum((pred[L] == k) & (ta[L] != k))
        fn = np.sum((pred[L] != k) & (ta[L] == k))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    rows_l = (tab["label"] >= 0)
    return dict(n_labelled=int(L.sum()), obs_top1=float(correct.sum() / max(L.sum(), 1)),
                macro_f1_trip=float(np.mean(f1s)) if f1s else np.nan,
                obs_log_loss=float(np.mean(-np.log(np.clip(p_true[L], 1e-6, 1)))),
                miss_rate=float(1 - hit[L].mean()),
                _y=tab["label"][rows_l], _p=probs_flat[rows_l])


def aggregate(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    w = df.n_labelled / df.n_labelled.sum()
    y = np.concatenate([r["_y"] for r in rows])
    p = np.concatenate([r["_p"] for r in rows])
    return dict(trips=len(df), points=int(df.n_labelled.sum()),
                obs_top1=float((df.obs_top1 * w).sum()), macro_f1_trip=float(df.macro_f1_trip.mean()),
                obs_log_loss=float((df.obs_log_loss * w).sum()), miss_rate=float((df.miss_rate * w).sum()),
                cand_macro_f1=float(f1_score(y, (p >= 0.5).astype(int), average="macro")),
                cand_log_loss=float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]))), df


def run_variant(name, tabs, folds, st_tabs):
    oof = {}
    fold_rows, hist = [], []
    t_cv = time.time()
    for k, val in enumerate(folds):
        tr = [t for j, f in enumerate(folds) if j != k for t in f]
        Xs, Ms, ys = zip(*[to_padded(tabs[t], train=True)[:3] for t in tr])
        X, M, y = np.concatenate(Xs), np.concatenate(Ms), np.concatenate(ys)
        Vs = {t: to_padded(tabs[t], train=True) for t in val}
        Xv = np.concatenate([v[0] for v in Vs.values()]); Mv = np.concatenate([v[1] for v in Vs.values()])
        yv = np.concatenate([v[2] for v in Vs.values()])
        s = ListwiseScorer(X.shape[2], seed=SEED + k).fit(X, M, y, Xv, Mv, yv)
        hist += [dict(fold=k, **h) for h in s.history]
        rows = []
        for t, (Xt, Mt, _, ptr, lab_ok) in Vs.items():
            Z = s.logits(Xt, Mt)
            oof[t] = (Z, Mt, ptr, lab_ok)
            rows.append(trip_metrics(tabs[t], ptr, unpad(s.proba_from_logits(Z, Mt, 1.0), ptr), lab_ok))
        agg, _ = aggregate(rows)
        fold_rows.append(dict(variant=name, fold=k, train_s=s.train_time, **agg))
        print(name, "fold", k, {kk: round(v, 4) for kk, v in agg.items() if isinstance(v, float)}, flush=True)
    cv_s = time.time() - t_cv
    # temperature on OOF (training trips only), then OOF metrics with calibration
    Zc = np.concatenate([o[0] for o in oof.values()]); Mc = np.concatenate([o[1] for o in oof.values()])
    yc = np.concatenate([to_padded(tabs[t], train=True)[2] for t in oof])
    T, nT, n1 = fit_temperature(Zc, Mc, yc)
    helper = ListwiseScorer(Zc.shape[1])
    cv_rows = [trip_metrics(tabs[t], o[2], unpad(helper.proba_from_logits(o[0], o[1], T), o[2]), o[3]) | {"trip": t}
               for t, o in oof.items()]
    cv_agg, cv_df = aggregate(cv_rows)
    # refit on all CV trips -> student test
    all_tr = sorted(sum(folds, []))
    Xs, Ms, ys = zip(*[to_padded(tabs[t], train=True)[:3] for t in all_tr])
    final = ListwiseScorer(Xs[0].shape[2], seed=SEED).fit(np.concatenate(Xs), np.concatenate(Ms), np.concatenate(ys))
    final.temperature = T
    st_rows, t_inf = [], 0.0
    for t, tab in st_tabs.items():
        X, M, _, ptr, lab_ok = to_padded(tab)
        t0 = time.time()
        P = final.predict_proba_padded(X, M)
        t_inf += time.time() - t0
        st_rows.append(trip_metrics(tab, ptr, unpad(P, ptr), lab_ok) | {"trip": t})
    st_agg, st_df = aggregate(st_rows)
    return dict(final=final, T=T, oof_nll=(n1, nT), cv=cv_agg, cv_df=cv_df, st=st_agg, st_df=st_df,
                folds=fold_rows, hist=hist, cv_s=cv_s, final_train_s=final.train_time, st_score_s=t_inf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default=os.path.join(ROOT, "data", "processed"))
    ap.add_argument("--features", default=os.path.join(ROOT, "data", "features"))
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    os.makedirs(ART, exist_ok=True)
    os.makedirs(os.path.join(HERE, "configs"), exist_ok=True)
    folds = load_folds()["folds"]
    cv_trips = sorted(sum(folds, []))
    st_trips = sorted(os.listdir(os.path.join(a.processed, "student_test")))
    t0 = time.time()
    p, cfg = fit_noise(a.processed, cv_trips)
    json.dump(cfg, open(os.path.join(HERE, "configs", "tracker_teamdata.json"), "w"), indent=1)
    print("noise fit", round(time.time() - t0, 1), "s ->", p, flush=True)

    def build(with_imm):
        tt = time.time()
        tr = dict(zip(cv_trips, Parallel(n_jobs=12)(delayed(trip_table)(a.processed, a.features, "train", t, p,
                                                                          REINIT_GAP_S, with_imm) for t in cv_trips)))
        st = dict(zip(st_trips, Parallel(n_jobs=10)(delayed(trip_table)(a.processed, a.features, "student_test", t,
                                                                          p, REINIT_GAP_S, with_imm) for t in st_trips)))
        return tr, st, time.time() - tt

    results = {}
    for name, with_imm in [("team_only", False), ("team_plus_imm", True)]:
        tr, st, feat_s = build(with_imm)
        r = run_variant(name, tr, folds, st)
        r["feat_s"] = feat_s
        r["names"] = tr[cv_trips[0]]["names"]
        results[name] = r
        torch.save(dict(state=r["final"].state(), feature_names=r["names"], tracker=p.__dict__,
                        reinit_gap_s=REINIT_GAP_S, kind=f"listwise_mlp_{name}"),
                   os.path.join(ART, f"{name}_teamdata.pt"))
        r["cv_df"].assign(variant=name).to_csv(os.path.join(RES, f"cv_per_trip_{name}.csv"), index=False)
        r["st_df"].assign(variant=name).to_csv(os.path.join(RES, f"student_test_per_trip_{name}.csv"), index=False)
    pd.DataFrame([f for r in results.values() for f in r["folds"]]).to_csv(os.path.join(RES, "cv_folds.csv"), index=False)
    pd.DataFrame([dict(variant=n, **h) for n, r in results.items() for h in r["hist"]]).to_csv(
        os.path.join(RES, "cv_history.csv"), index=False)
    summ = []
    for n, r in results.items():
        f = pd.DataFrame(r["folds"])
        summ.append(dict(variant=n, split="cv_5fold_oof", **r["cv"], fold_top1_mean=f.obs_top1.mean(),
                         fold_top1_std=f.obs_top1.std(), temperature=r["T"], cv_wall_s=r["cv_s"], feature_s=r["feat_s"]))
        summ.append(dict(variant=n, split="student_test", **r["st"], final_train_s=r["final_train_s"],
                         scoring_s=r["st_score_s"]))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(RES, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
