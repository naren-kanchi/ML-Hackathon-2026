"""Bayesian hyperparameter optimisation of the LightGBM LambdaRank candidate ranker
(src/models/lgbm_ranker.py), selected after the 5-model comparison
(experiments/results/common_eval/summary.csv: best CV Top-1 after HMM/Viterbi).

Sampler: Optuna TPE (sequential model-based Bayesian optimisation), seed 20261118.
Data: the 78 training trips in the 5 frozen trip-grouped folds
(outputs/splits/folds.json). Student-test is used once, after the search, for the
tuned-vs-untuned comparison -- never inside the search.

Stage A -- ranker hyperparameters
  objective: mean over folds of out-of-fold observation-level Top-1
             (correct / team-labelled points of the fold; points whose true arc is
             not a candidate, or with no candidates, count as wrong)
  n_estimators       int[100, 800]
  learning_rate      logU[0.01, 0.3]
  num_leaves         int[8, 255]
  min_child_samples  int[5, 200]
  subsample          U[0.5, 1.0]   (bagging every iteration)
  colsample_bytree   U[0.4, 1.0]
  reg_lambda         logU[1e-3, 30]
  reg_alpha          logU[1e-3, 10]
  min_split_gain     U[0, 1]
  trial 0 = the untuned configuration from lgbm_ranker.CONFIG.
Stage B -- HMM/Viterbi decoder (pathsnap/hmm.py) on the stage-A winner's
  calibrated OOF probabilities; objective = mean fold OOF Top-1 AFTER Viterbi.
  beta logU[1, 30] m, w_emit U[0.3, 3], v_max U[30, 80] m/s, back_tol U[5, 30] m,
  gap_break_s U[30, 300] s; trial 0 = default HMMParams().
Not searched: candidate radius / count (fixed by src/features/build_features.py,
rebuilding every feature table per trial is too slow), and the softmax temperature
(fitted on OOF scores, as in lgbm_ranker.py).

Time budget (30 min) run: 15 stage-A trials, 6 stage-B trials. The untuned
row reuses experiments/results/viterbi_shared/lgbm_ranker (same folds, temperature
scheme, default decoder and metrics) instead of decoding it again.

python -m src.optimization.bayes_opt_lgbm [--trials-a 15] [--trials-b 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)   # lgbm_ranker uses repo-relative paths
import src.models.lgbm_ranker as L  # noqa: E402
from pathsnap.evaluate import summarise  # noqa: E402
from pathsnap.hmm import HMMParams  # noqa: E402
from pathsnap.splits import SEED, load_folds  # noqa: E402
from src.models.imm_2ekf.teamdata import decode_team_trip  # noqa: E402

RES = os.path.join(ROOT, "experiments", "results", "bayes_opt_lgbm")
FOLDS = [sorted(f) for f in load_folds()["folds"]]
CV = sorted(t for f in FOLDS for t in f)
FOLD = {t: k for k, f in enumerate(FOLDS) for t in f}
SEARCH_KEYS = ["n_estimators", "learning_rate", "num_leaves", "min_child_samples", "subsample",
               "colsample_bytree", "reg_lambda", "reg_alpha", "min_split_gain"]
UNTUNED = {**{k: L.CONFIG[k] for k in SEARCH_KEYS if k in L.CONFIG}, "reg_lambda": 0.0, "reg_alpha": 0.0,
           "min_split_gain": 0.0}


def fit(D, trips, params, n_jobs=-1):
    m = D.rows(trips) & D.labelled
    keep = m & D.group_has_true[D.gid]
    _, sizes = np.unique(D.gid[keep], return_counts=True)
    cfg = {k: v for k, v in L.CONFIG.items() if k != "eval_at"}
    cfg.update(params)
    cfg["n_jobs"] = n_jobs
    model = lgb.LGBMRanker(**cfg)
    t0 = time.perf_counter()
    model.fit(D.X[keep], D.y[keep], group=sizes, eval_at=L.CONFIG["eval_at"])
    return model, time.perf_counter() - t0


def oof_scores(D, params):
    """5 folds trained concurrently (threads; LightGBM releases the GIL), 3 cores each."""
    res = Parallel(n_jobs=5, prefer="threads")(
        delayed(fit)(D, [t for t in CV if t not in val], params, 3) for val in FOLDS)
    oof = np.full(len(D.y), np.nan)
    fold_of = np.full(len(D.y), -1)
    for k, (val, (model, _)) in enumerate(zip(FOLDS, res)):
        m = D.rows(val)
        oof[m] = model.predict(D.X[m])
        fold_of[m] = k
    return oof, fold_of, [s for _, s in res]


def fold_top1(D, score, fold_of):
    """Per-fold Top-1 with the full labelled-point denominator."""
    top = L.group_top1(np.where(np.isnan(score), -np.inf, score), D.gid)
    hit = top & (D.y == 1)
    out = []
    for k, val in enumerate(FOLDS):
        n = sum(D.n_labelled_points[t] for t in val)
        out.append(float(hit[fold_of == k].sum() / n))
    return out


def calibrated(D, oof, fold_of):
    """Per-fold temperature fit on the OTHER folds' OOF scores (lgbm_ranker scheme)."""
    cvm = D.rows(CV) & D.labelled
    proba = np.full(len(D.y), np.nan)
    for k in range(len(FOLDS)):
        other = cvm & (fold_of != k)
        Tk = L.fit_temperature(oof[other], D.y[other], D.gid[other])
        m = fold_of == k
        proba[m] = L.group_softmax(oof[m], D.gid[m], Tk)
    return proba, L.fit_temperature(oof[cvm], D.y[cvm], D.gid[cvm])


def tables(D, proba, trips):
    return {t: pd.DataFrame({"pt": D.pt[D.trip == t], "arc_id": D.arc[D.trip == t], "proba": proba[D.trip == t]})
            for t in trips}


def decode(part, tabs, hp):
    out = Parallel(n_jobs=14)(delayed(decode_team_trip)(part, t, None, tab, hp) for t, tab in tabs.items())
    return pd.DataFrame([r for o in out for r in o])


def fold_mean(df, mode="hmm"):
    d = df[df["mode"] == mode].copy()
    d["fold"] = d.trip.astype(str).str.zfill(8).map(FOLD)
    return float(d.groupby("fold").apply(lambda g: np.average(g.top1, weights=g.n_fix), include_groups=False).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials-a", type=int, default=15)
    ap.add_argument("--trials-b", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    tr, st = L.Part("train"), L.Part("student_test")

    # ---------------- stage A
    def obj_a(trial):
        p = dict(n_estimators=trial.suggest_int("n_estimators", 100, 800),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 num_leaves=trial.suggest_int("num_leaves", 8, 255),
                 min_child_samples=trial.suggest_int("min_child_samples", 5, 200),
                 subsample=trial.suggest_float("subsample", 0.5, 1.0),
                 colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
                 reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                 reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                 min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0))
        t0 = time.time()
        oof, fold_of, secs = oof_scores(tr, p)
        f = fold_top1(tr, oof, fold_of)
        for k, v in enumerate(f):
            trial.set_user_attr(f"fold{k}_top1", v)
        trial.set_user_attr("fit_s", sum(secs))
        print(f"A {trial.number:3d}: {np.mean(f):.4f}  ({time.time() - t0:.0f}s) {p}", flush=True)
        return float(np.mean(f))

    t0 = time.time()
    A = optuna.create_study(direction="maximize", study_name="lgbm_stageA",
                            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=10))
    # trial 0 = untuned; reg/min_split_gain default 0 is outside the log ranges -> use lower bounds
    A.enqueue_trial({**UNTUNED, "reg_lambda": 1e-3, "reg_alpha": 1e-3})
    A.optimize(obj_a, n_trials=a.trials_a)
    A.trials_dataframe().to_csv(os.path.join(RES, "stageA_trials.csv"), index=False)
    best_p, ta = A.best_params, time.time() - t0
    print("stage A best", A.best_value, best_p, flush=True)

    # ---------------- stage B
    oof, fold_of, _ = oof_scores(tr, best_p)
    proba_cv, _ = calibrated(tr, oof, fold_of)
    tabs_cv = tables(tr, proba_cv, CV)

    decoded = {}

    def obj_b(trial):
        hp = HMMParams(beta=trial.suggest_float("beta", 1.0, 30.0, log=True),
                       w_emit=trial.suggest_float("w_emit", 0.3, 3.0),
                       v_max=trial.suggest_float("v_max", 30.0, 80.0),
                       back_tol=trial.suggest_float("back_tol", 5.0, 30.0),
                       gap_break_s=trial.suggest_float("gap_break_s", 30.0, 300.0))
        t1 = time.time()
        df = decode("train", tabs_cv, hp)
        decoded[trial.number] = df
        v = fold_mean(df)
        print(f"B {trial.number:3d}: {v:.4f}  ({time.time() - t1:.0f}s) {asdict(hp)}", flush=True)
        return v

    t1 = time.time()
    B = optuna.create_study(direction="maximize", study_name="lgbm_stageB_decoder",
                            sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=6))
    d0 = asdict(HMMParams())
    B.enqueue_trial({k: d0[k] for k in ("beta", "w_emit", "v_max", "back_tol", "gap_break_s")})
    B.optimize(obj_b, n_trials=a.trials_b)
    B.trials_dataframe().to_csv(os.path.join(RES, "stageB_trials.csv"), index=False)
    best_hp, tb = HMMParams(**B.best_params), time.time() - t1
    print("stage B best", B.best_value, best_hp, flush=True)

    # ---------------- final: untuned vs tuned (identical protocol)
    rows = []
    for name, p, hp in (("untuned", UNTUNED, HMMParams()), ("tuned", best_p, best_hp)):
        oof, fold_of, secs = oof_scores(tr, p)
        f_clf = fold_top1(tr, oof, fold_of)
        proba_cv, T = calibrated(tr, oof, fold_of)
        final, fit_s = fit(tr, CV, p)
        t2 = time.perf_counter()
        s_st = final.predict(st.X)
        p_st = L.group_softmax(s_st, st.gid, T)
        inf_s = time.perf_counter() - t2
        final.booster_.save_model(os.path.join(RES, f"lgbm_ranker_{name}.txt"))
        if name == "untuned":   # identical protocol already decoded in viterbi_shared/lgbm_ranker
            prev = os.path.join(ROOT, "experiments", "results", "viterbi_shared", "lgbm_ranker")
            dec_cv = pd.read_csv(os.path.join(prev, "cv_per_trip.csv"), dtype={"trip": str})
            dec_st = pd.read_csv(os.path.join(prev, "student_test_per_trip.csv"), dtype={"trip": str})
        else:                   # CV decode of the best stage-B trial (same probabilities, same hp)
            dec_cv = decoded[B.best_trial.number]
            dec_st = decode("student_test", tables(st, p_st, sorted(np.unique(st.trip))), hp)
        dec_cv.to_csv(os.path.join(RES, f"cv_per_trip_{name}.csv"), index=False)
        dec_st.to_csv(os.path.join(RES, f"student_test_per_trip_{name}.csv"), index=False)
        for split, D, prob, dec in (("cv_5fold_oof", tr, proba_cv, dec_cv), ("student_test", st, p_st, dec_st)):
            m = D.labelled & ~np.isnan(prob) & D.group_has_true[D.gid]
            nll = L.group_nll(np.log(np.clip(prob[m], 1e-15, 1)), D.y[m], D.gid[m], 1.0)
            s = summarise(dec).set_index("mode")
            rows.append(dict(model=name, split=split,
                             top1_clf=s.loc["clf", "top1"], macro_f1_clf=s.loc["clf", "macro_f1_tripmean"],
                             top1_hmm=s.loc["hmm", "top1"], macro_f1_hmm=s.loc["hmm", "macro_f1_tripmean"],
                             continuity_hmm=s.loc["hmm", "continuity"], route_f1_trim_hmm=s.loc["hmm", "route_f1_trim"],
                             log_loss_covered=nll,
                             fold_top1_clf_sd=float(np.std(f_clf, ddof=1)) if split.startswith("cv") else np.nan,
                             temperature=T, cv_fit_s=sum(secs), final_fit_s=fit_s,
                             infer_us_per_row=1e6 * inf_s / len(st.y)))
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(RES, "tuned_vs_untuned.csv"), index=False)
    json.dump(dict(model="lgbm_ranker (LightGBM LambdaRank)", sampler="optuna TPESampler(seed=20261118)",
                   optuna_version=optuna.__version__, lightgbm_version=lgb.__version__,
                   stageA=dict(objective="mean 5-fold OOF observation Top-1, classifier-only", trials=a.trials_a,
                               wall_s=ta, untuned_value=A.trials[0].value, best_value=A.best_value,
                               best=best_p, untuned=UNTUNED),
                   stageB=dict(objective="mean 5-fold OOF observation Top-1 after HMM/Viterbi", trials=a.trials_b,
                               wall_s=tb, untuned_value=B.trials[0].value, best_value=B.best_value,
                               best=asdict(best_hp))),
              open(os.path.join(RES, "search_info.json"), "w"), indent=1)
    pd.set_option("display.width", 250)
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
