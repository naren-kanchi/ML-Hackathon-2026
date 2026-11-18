"""Run the shared HMM/Viterbi decoder (pathsnap/hmm.py) on models that had no
decoded results yet, with the shared labels and metrics:

  hgb             the team's HistGradientBoosting (src/models/histgradientboosting),
                  kept as designed: its own candidates (arc midpoints within 80 m,
                  <= 40), its 13 features, its tuned parameters and its own
                  training labels. CV: refit per frozen fold on the fold's
                  training trips (a model fit on all trips cannot score training
                  trips fairly). Student-test: refit on all 80 training trips,
                  like the committed model. Refit locally because the committed
                  joblib was pickled with scikit-learn 1.9.1 (this machine: 1.7.2).
                  Emissions: positive-class probability normalised over each
                  fix's candidates (as backend/services/map_matching.py does).
  imm_teamdata    src/models/imm_2ekf on the team preprocessing, both variants
                  (team_only, team_plus_imm), from saved OOF / student-test
                  probabilities; decoded on the team's cleaned, thinned points.

Labels: hgb uses the shared derived labels (outputs/labels, all fixes; misses
count as errors). imm_teamdata uses the team's true_arc on kept points
(denominator = labelled points), matching its classifier-only table.
Outputs: experiments/results/viterbi_shared/<model>/

  logreg          the team's Logistic Regression (same features/candidates/labels as hgb,
                  C=1.0, StandardScaler), refit per frozen fold / on all 80 trips.
  lgbm            the team's LightGBM LambdaRank on the team preprocessing (imported).

python scripts/viterbi_others.py [hgb] [logreg] [imm_teamdata] [lgbm]
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.candidates import Candidates  # noqa: E402
from pathsnap.context import TripContext  # noqa: E402
from pathsnap.evaluate import evaluate_probs, summarise  # noqa: E402
from pathsnap.hmm import HMMParams, classifier_only, reconstruct_route, viterbi  # noqa: E402
from pathsnap.io import DATA_ROOT, Trip, list_trips, load_trip  # noqa: E402
from pathsnap.labels import load_route  # noqa: E402
from pathsnap.metrics import continuity, macro_f1_trip, route_metrics  # noqa: E402
from pathsnap.roadnet import RoadNet  # noqa: E402
from pathsnap.splits import load_folds  # noqa: E402
from src.models.imm_2ekf.teamdata import ctx_from_candidate_lists  # noqa: E402

OUT = os.path.join(ROOT, "experiments", "results", "viterbi_shared")


# ------------------------------------------------------------------ HGB
def _hgb_module():
    p = os.path.join(ROOT, "src", "models", "histgradientboosting", "train_histgradientboosting.py")
    spec = importlib.util.spec_from_file_location("hgb_train", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _logreg_module():
    p = os.path.join(ROOT, "src", "models", "logistic_regression", "train_logistic_regression.py")
    spec = importlib.util.spec_from_file_location("logreg_train", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _hgb_eval_trip(model, part, tid, save_dir=None):
    sys.path.insert(0, ROOT)
    from backend.services.map_matching import extract_trip_features
    _, feats, meta = extract_trip_features(Path(DATA_ROOT) / part / tid)
    probs = []
    for F in feats:
        s = model.predict_proba(F)[:, list(model.classes_).index(1)]
        probs.append(s / s.sum() if s.sum() > 0 else np.full(len(s), 1 / len(s)))
    trip = load_trip(part, tid)
    ctx, pr = ctx_from_candidate_lists(trip, RoadNet(trip), meta["candidate_arcs_by_observation"], probs)
    r = evaluate_probs(part, tid, pr, ctx=ctx)
    if save_dir:   # per-raw-fix decoded arcs, for the common-set comparison
        os.makedirs(save_dir, exist_ok=True)
        np.savez_compressed(os.path.join(save_dir, f"{part}_{tid}.npz"), clf=classifier_only(ctx, pr)["arc"],
                            hmm=viterbi(ctx, pr)["arc"])
    return [r["clf"], r["hmm"]]


def run_hgb():
    from sklearn.ensemble import HistGradientBoostingClassifier
    params = dict(loss="log_loss", learning_rate=0.03846173685406504, max_iter=210, max_leaf_nodes=60,
                  l2_regularization=9.476210031010524, min_samples_leaf=10, random_state=42)
    run_midpoint("histgradientboosting", _hgb_module(), lambda: HistGradientBoostingClassifier(**params), params)


def run_logreg():
    m = _logreg_module()
    run_midpoint("logistic_regression", m, lambda: m.build_model(C=1.0),
                 dict(C=1.0, scaler="StandardScaler", solver="lbfgs", max_iter=1000, random_state=42))


def run_midpoint(name, m, factory, params):
    """Team models built on src/models/histgradientboosting-style features (arc midpoints,
    80 m / 40 candidates, labels projected to the nearest GPS sample)."""
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    arc_dir = os.path.join(ROOT, "outputs", "preds", f"arcs_{name}")
    folds = load_folds()["folds"]
    cache = {}

    def rows(tids):
        X, y = [], []
        for t in tids:
            if t not in cache:
                x_, y_, _ = m.features_for_trip(Path(DATA_ROOT) / "train" / t)
                cache[t] = (x_, y_)
            X.append(cache[t][0]); y.append(cache[t][1])
        return np.vstack(X), np.concatenate(y)
    res, timing = [], []
    for k, val in enumerate(folds):
        tr = [t for j, f in enumerate(folds) if j != k for t in f]
        X, y = rows(tr)
        t0 = time.time()
        model = factory().fit(X, y)
        timing.append(dict(fold=k, fit_s=time.time() - t0, rows=len(y)))
        out = Parallel(n_jobs=8)(delayed(_hgb_eval_trip)(model, "train", t, arc_dir) for t in val)
        res += [r for o in out for r in o]
        print(name, "fold", k, "done", flush=True)
    X, y = rows(list_trips("train"))
    t0 = time.time()
    final = factory().fit(X, y)
    timing.append(dict(fold="all80", fit_s=time.time() - t0, rows=len(y)))
    out = Parallel(n_jobs=8)(delayed(_hgb_eval_trip)(final, "student_test", t, arc_dir)
                             for t in list_trips("student_test"))
    st = [r for o in out for r in o]
    _write(d, pd.DataFrame(res), pd.DataFrame(st), dict(params=params, timing=timing,
                                                      note="team features/candidates/labels for training; shared "
                                                           "labels, decoder and metrics for evaluation"))


# ------------------------------------------------------------------ IMM on team data
def _imm_team_trip(part, tid, probs_path, table=None, hp=None):
    from src.models.imm_2ekf.teamdata import decode_team_trip
    return decode_team_trip(part, tid, None if probs_path is None else np.load(probs_path), table, hp)


def run_imm_teamdata():
    folds = load_folds()["folds"]
    cv_trips = sorted(sum(folds, []))
    for v in ("team_only", "team_plus_imm"):
        P = os.path.join(ROOT, "outputs", "preds", f"imm_teamdata_{v}")
        d = os.path.join(OUT, f"imm_teamdata_{v}")
        os.makedirs(d, exist_ok=True)
        cv = Parallel(n_jobs=10)(delayed(_imm_team_trip)("train", t, os.path.join(P, f"oof_{t}.npy")) for t in cv_trips)
        st = Parallel(n_jobs=10)(delayed(_imm_team_trip)("student_test", t, os.path.join(P, f"st_{t}.npy"))
                                 for t in list_trips("student_test"))
        _write(d, pd.DataFrame([r for o in cv for r in o]), pd.DataFrame([r for o in st for r in o]),
               dict(note="team-labelled points only (true_arc >= 0); probabilities from "
                         "src/models/imm_2ekf/train_imm_teamdata.py (calibrated OOF / refit)"))


def run_lgbm():
    """Team LightGBM LambdaRank (src/models/lgbm_ranker.py) refit with its own code and
    folds-file scheme: OOF scores per frozen fold, per-fold temperature fit on the other
    folds' OOF, final fit on the 78 CV trips + global temperature for student-test.
    Imported, not executed, so the team's committed reports are not rewritten."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    import src.models.lgbm_ranker as L
    folds = [sorted(f) for f in load_folds()["folds"]]
    cv_trips = sorted(t for f in folds for t in f)
    tr, st = L.Part("train"), L.Part("student_test")
    oof = np.full(len(tr.y), np.nan)
    fold_of = np.full(len(tr.y), -1)
    timing = []
    for k, val in enumerate(folds):
        model, secs, _ = L.fit_ranker(tr, [t for t in cv_trips if t not in val])
        m = tr.rows(val)
        oof[m] = model.predict(tr.X[m])
        fold_of[m] = k
        timing.append(dict(fold=k, fit_s=secs))
        print("lgbm fold", k, round(secs, 1), "s", flush=True)
    cvm = tr.rows(cv_trips) & tr.labelled
    proba = np.full(len(tr.y), np.nan)
    for k in range(len(folds)):
        other = cvm & (fold_of != k)
        Tk = L.fit_temperature(oof[other], tr.y[other], tr.gid[other])
        m = fold_of == k
        proba[m] = L.group_softmax(oof[m], tr.gid[m], Tk)
    T = L.fit_temperature(oof[cvm], tr.y[cvm], tr.gid[cvm])
    final, secs, _ = L.fit_ranker(tr, cv_trips)
    timing.append(dict(fold="final", fit_s=secs, T=T))
    p_st = L.group_softmax(final.predict(st.X), st.gid, T)
    os.chdir(cwd)
    tab = lambda D, p, t: pd.DataFrame({"pt": D.pt[D.trip == t], "arc_id": D.arc[D.trip == t],  # noqa: E731
                                        "proba": p[D.trip == t]})
    d = os.path.join(OUT, "lgbm_ranker")
    os.makedirs(d, exist_ok=True)
    pdir = os.path.join(ROOT, "outputs", "preds", "lgbm_ranker")
    os.makedirs(pdir, exist_ok=True)
    for t in cv_trips:
        tab(tr, proba, t).to_parquet(os.path.join(pdir, f"oof_{t}.parquet"), index=False)
    for t in list_trips("student_test"):
        tab(st, p_st, t).to_parquet(os.path.join(pdir, f"st_{t}.parquet"), index=False)
    cv = Parallel(n_jobs=10)(delayed(_imm_team_trip)("train", t, None, tab(tr, proba, t)) for t in cv_trips)
    sv = Parallel(n_jobs=10)(delayed(_imm_team_trip)("student_test", t, None, tab(st, p_st, t))
                             for t in list_trips("student_test"))
    _write(d, pd.DataFrame([r for o in cv for r in o]), pd.DataFrame([r for o in sv for r in o]),
           dict(config=L.CONFIG, timing=timing, note="team-labelled points only; team code imported"))


def _write(d, cv, st, info):
    cv.to_csv(os.path.join(d, "cv_per_trip.csv"), index=False)
    st.to_csv(os.path.join(d, "student_test_per_trip.csv"), index=False)
    a, b = summarise(cv), summarise(st)
    a.insert(0, "split", "cv_oof")
    b.insert(0, "split", "student_test")
    s = pd.concat([a, b])
    s.to_csv(os.path.join(d, "summary.csv"), index=False)
    json.dump(info, open(os.path.join(d, "run_info.json"), "w"), indent=1, default=str)
    print(d)
    print(s[["split", "mode", "trips", "fixes", "top1", "macro_f1_tripmean", "continuity", "route_f1",
             "route_f1_trim", "route_lcs", "restarts_total"]].round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["hgb", "logreg", "imm_teamdata", "lgbm"]
    if "hgb" in which:
        run_hgb()
    if "logreg" in which:
        run_logreg()
    if "lgbm" in which:
        run_lgbm()
    if "imm_teamdata" in which:
        run_imm_teamdata()
