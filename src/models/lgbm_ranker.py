"""
PathSnap model #5 - LightGBM LambdaRank candidate ranker.

Each GPS point has a small group of candidate arcs; exactly one is the true arc.
Instead of classifying rows independently, LambdaRank learns a score that ranks the
candidates inside each point's group (query = (trip, pt)) so the true arc comes first.

Adaptation to probabilities: p = softmax(score / T) within each point's group, with
one temperature T fitted on TRAIN out-of-fold scores (multiclass log loss).

Data rules
  * features = EMISSION_FEATURES only (src.features.build_features); never labels/route.
  * folds    = the shared team folds in outputs/splits/folds.json (5 trip-grouped folds).
               Trips listed there as label-unreliable are excluded from training and CV.
  * student_test is used for evaluation only (no training, T fitting or choices on it).
  * training drops points whose true arc is not a candidate (no ranking signal);
    evaluation keeps them (they count as wrong for Top-1).

Usage (from the repo root):
    python -m src.models.lgbm_ranker
"""
from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime, timezone

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.optimize import minimize_scalar
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)

from src.features.build_features import EMISSION_FEATURES, load_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODEL = "lgbm_ranker"
FAMILY = "Boosting - learning to rank (LambdaRank)"
SEED = 42
FEATURES_DIR, PROCESSED_DIR = "data/features", "data/processed"
SCORES_DIR = "data/scores"
FOLDS_PATH = "outputs/splits/folds.json"
REPORT_DIR = os.path.join("reports", "models", MODEL)
ARTIFACT_DIR = os.path.join("artifacts", "models")
CONFIG = dict(objective="lambdarank", metric="ndcg", eval_at=[1], n_estimators=500, learning_rate=0.05,
              num_leaves=31, min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
              random_state=SEED, n_jobs=-1, verbose=-1)
T_BOUNDS = (0.05, 20.0)
P_CLIP = 1e-15

# chart tokens (reference palette, light mode)
C_BLUE, C_INK, C_INK2, C_MUTED, C_GRID, C_SURF = "#2a78d6", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ_BLUES = ["#f0f6fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
class Part:
    """All candidate rows of one partition, sorted by (trip, pt) with a group id per point."""

    def __init__(self, part: str):
        X, y, trip, meta = load_dataset(FEATURES_DIR, part, "emission", labelled_only=False)
        df = pd.concat([meta[["trip", "pt", "arc_id"]], X, y.rename("label")], axis=1)
        df = df.sort_values(["trip", "pt"], kind="mergesort").reset_index(drop=True)
        self.part = part
        self.X = df[EMISSION_FEATURES].astype(np.float32)
        self.y = df.label.to_numpy()
        self.trip = df.trip.to_numpy()
        self.pt = df.pt.to_numpy()
        self.arc = df.arc_id.to_numpy()
        new = np.r_[True, (self.trip[1:] != self.trip[:-1]) | (self.pt[1:] != self.pt[:-1])]
        self.gid = np.cumsum(new) - 1
        self.gstart = np.flatnonzero(new)
        self.labelled = self.y >= 0
        pos = np.zeros(self.gid.max() + 1, bool)
        pos[self.gid[self.y == 1]] = True
        self.group_has_true = pos
        self.n_labelled_points = labelled_point_counts(part)

    def rows(self, trips):
        return np.isin(self.trip, list(trips))


def labelled_point_counts(part: str) -> dict:
    """Labelled GPS points per trip (incl. points with no candidates at all)."""
    d = os.path.join(PROCESSED_DIR, part)
    return {t: int((pd.read_parquet(os.path.join(d, t, "points.parquet"), columns=["true_arc"]).true_arc >= 0).sum())
            for t in sorted(os.listdir(d)) if os.path.isdir(os.path.join(d, t))}


# ----------------------------------------------------------------------------
# group-wise helpers (rows are contiguous per group)
# ----------------------------------------------------------------------------
def _starts(gid):
    return np.flatnonzero(np.r_[True, gid[1:] != gid[:-1]])


def group_softmax(score, gid, T):
    s = score / T
    st = _starts(gid)
    idx = np.repeat(np.arange(len(st)), np.diff(np.r_[st, len(s)]))
    e = np.exp(s - np.maximum.reduceat(s, st)[idx])
    return e / np.add.reduceat(e, st)[idx]


def group_top1(score, gid):
    """Row mask: highest-scoring candidate of each group (ties -> first row, i.e. nearest)."""
    order = np.lexsort((-score, gid))
    first = order[np.r_[True, gid[order][1:] != gid[order][:-1]]]
    m = np.zeros(len(score), bool)
    m[first] = True
    return m


def group_nll(score, y, gid, T):
    """Multiclass log loss of the softmax over groups that contain the true arc."""
    keep = np.isin(gid, gid[y == 1])
    s, yy, g = score[keep] / T, y[keep], gid[keep]
    st = _starts(g)
    lse = np.log(np.add.reduceat(np.exp(s - np.maximum.reduceat(s, st)[np.repeat(np.arange(len(st)), np.diff(np.r_[st, len(s)]))]), st)) \
        + np.maximum.reduceat(s, st)
    return float(np.mean(lse - s[yy == 1]))


def fit_temperature(score, y, gid) -> float:
    r = minimize_scalar(lambda T: group_nll(score, y, gid, T), bounds=T_BOUNDS, method="bounded")
    return float(r.x)


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
def fit_ranker(D: Part, trips):
    m = D.rows(trips) & D.labelled
    lab_groups = np.unique(D.gid[m])
    keep = m & D.group_has_true[D.gid]
    dropped = int(len(lab_groups) - len(np.unique(D.gid[keep])))
    _, sizes = np.unique(D.gid[keep], return_counts=True)
    model = lgb.LGBMRanker(**{k: v for k, v in CONFIG.items() if k != "eval_at"})
    t0 = time.perf_counter()
    model.fit(D.X[keep], D.y[keep], group=sizes, eval_at=CONFIG["eval_at"])
    return model, time.perf_counter() - t0, dict(train_rows=int(keep.sum()), train_groups=int(len(sizes)),
                                                 dropped_groups_no_true=dropped)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def candidate_metrics(D: Part, mask, score, proba) -> dict:
    """mask selects labelled rows of the trips being evaluated."""
    y, g, s, p = D.y[mask], D.gid[mask], score[mask], proba[mask]
    top = group_top1(s, g).astype(int)
    thr = (p >= 0.5).astype(int)
    trips = np.unique(D.trip[mask])
    n_pts = sum(D.n_labelled_points[t] for t in trips)
    groups_true = np.unique(g[y == 1])
    out = {"n_rows": int(mask.sum()), "n_groups": int(len(np.unique(g))), "n_labelled_points": n_pts,
           "obs_top1": float(((top == 1) & (y == 1)).sum() / n_pts),
           "candidate_miss_rate": float(1 - len(groups_true) / n_pts),
           "logloss": float(log_loss(y, np.clip(p, P_CLIP, 1 - P_CLIP), labels=[0, 1])),
           "group_nll": None, "roc_auc": float(roc_auc_score(y, p))}
    for name, pred in (("top1", top), ("thr05", thr)):
        out.update({f"{name}_accuracy": float(accuracy_score(y, pred)),
                    f"{name}_precision": float(precision_score(y, pred, zero_division=0)),
                    f"{name}_recall": float(recall_score(y, pred, zero_division=0)),
                    f"{name}_f1": float(f1_score(y, pred, zero_division=0))})
    return out


# ----------------------------------------------------------------------------
# plots
# ----------------------------------------------------------------------------
def _style(ax):
    ax.set_facecolor(C_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=C_INK2, labelsize=9)
    ax.title.set_color(C_INK)


def plot_confusion(y, pred, title, path):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    pct = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150, facecolor=C_SURF)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ_BLUES)
    ax.imshow(pct, cmap=cmap, vmin=0, vmax=100)
    for i in range(2):
        for j in range(2):
            ink = "#ffffff" if pct[i, j] > 55 else C_INK
            ax.text(j, i, f"{cm[i, j]:,}\n{pct[i, j]:.1f}% of row", ha="center", va="center", color=ink, fontsize=10)
    ax.set_xticks([0, 1], ["Predicted: other arc", "Predicted: true arc"])
    ax.set_yticks([0, 1], ["Actual: other arc", "Actual: true arc"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=C_INK2, labelsize=9, length=0)
    ax.set_title(title, fontsize=10.5, color=C_INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=C_SURF, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_calibration(y, p, path):
    frac, mean_p = calibration_curve(y, p, n_bins=10, strategy="uniform")
    counts = np.histogram(p, bins=np.linspace(0, 1, 11))[0]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(5.2, 5.6), dpi=150, facecolor=C_SURF,
                                  gridspec_kw={"height_ratios": [3, 1.1]}, sharex=True)
    _style(ax); _style(ax2)
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, ls=(0, (4, 3)))
    ax.text(0.62, 0.55, "perfect calibration", color=C_MUTED, fontsize=8.5, rotation=38)
    ax.plot(mean_p, frac, color=C_BLUE, lw=2, marker="o", ms=6, mec=C_SURF, mew=1.5)
    ax.set_ylabel("Observed share that is the true arc", color=C_INK2, fontsize=9)
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    ax.grid(color=C_GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Reliability of softmax probabilities - student_test (10 bins)", fontsize=10.5, loc="left")
    ax2.bar(np.linspace(0.05, 0.95, 10), counts, width=0.096, color=C_BLUE)
    ax2.set_yscale("log")
    ax2.set_ylabel("Candidates\n(log)", color=C_INK2, fontsize=9)
    ax2.set_xlabel("Predicted probability (bin)", color=C_INK2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, facecolor=C_SURF)
    plt.close(fig)


def plot_importance(booster, path):
    imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name())
    imp = (imp / imp.sum() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=150, facecolor=C_SURF)
    _style(ax)
    ax.barh(imp.index, imp.values, color=C_BLUE, height=0.62)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=8, color=C_INK2)
    ax.set_xlabel("Share of total split gain (%)", color=C_INK2, fontsize=9)
    ax.grid(axis="x", color=C_GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_title("Feature importance (gain) - final model", fontsize=10.5, loc="left")
    fig.tight_layout()
    fig.savefig(path, facecolor=C_SURF)
    plt.close(fig)


# ----------------------------------------------------------------------------
# outputs
# ----------------------------------------------------------------------------
def write_scores(D: Part, score, proba, trips):
    d = os.path.join(SCORES_DIR, MODEL, D.part)
    os.makedirs(d, exist_ok=True)
    for t in trips:
        m = D.trip == t
        pd.DataFrame({"pt": D.pt[m], "arc_id": D.arc[m], "score": score[m], "proba": proba[m]}).to_parquet(
            os.path.join(d, f"{t}.parquet"), index=False)


def versions():
    return {"python": platform.python_version(), "lightgbm": lgb.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__}


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    np.random.seed(SEED)
    with open(FOLDS_PATH) as fh:
        F = json.load(fh)
    folds = [sorted(f) for f in F["folds"]]
    excluded = sorted(F.get("excluded_label_unreliable", []))
    cv_trips = sorted(t for f in folds for t in f)

    tr, st = Part("train"), Part("student_test")
    assert set(tr.trip) == set(cv_trips) | set(excluded), "train trips do not match the shared folds"
    print(f"train: {len(tr.y):,} rows / {tr.gid.max() + 1:,} points; student_test: {len(st.y):,} rows; "
          f"CV trips {len(cv_trips)}, excluded {excluded}")

    # ---------------- cross-validation ----------------
    oof = np.full(len(tr.y), np.nan)
    fold_of = np.full(len(tr.y), -1)
    fold_info = []
    for k, val in enumerate(folds):
        trn = [t for t in cv_trips if t not in val]
        model, secs, info = fit_ranker(tr, trn)
        m = tr.rows(val)
        oof[m] = model.predict(tr.X[m])
        fold_of[m] = k
        fold_info.append({"fold": k, "train_trips": len(trn), "val_trips": len(val), "train_time_s": secs, **info})
        print(f"fold {k}: trained on {len(trn)} trips ({info['train_groups']:,} groups, "
              f"dropped {info['dropped_groups_no_true']} without true arc) in {secs:.1f}s", flush=True)

    cvm = tr.rows(cv_trips) & tr.labelled
    T_final = fit_temperature(oof[cvm], tr.y[cvm], tr.gid[cvm])
    proba_tr = np.full(len(tr.y), np.nan)
    fold_rows = []
    for k, val in enumerate(folds):
        other = cvm & (fold_of != k)
        T_k = fit_temperature(oof[other], tr.y[other], tr.gid[other])      # T never sees fold k
        m = fold_of == k
        proba_tr[m] = group_softmax(oof[m], tr.gid[m], T_k)
        ev = m & tr.labelled
        r = candidate_metrics(tr, ev, oof, proba_tr)
        r["group_nll"] = group_nll(oof[ev], tr.y[ev], tr.gid[ev], T_k)
        fold_rows.append({**fold_info[k], "temperature": T_k, **r})
    cv = pd.DataFrame(fold_rows)
    cv.to_csv(os.path.join(REPORT_DIR, "cv_folds.csv"), index=False)
    print(f"temperature (all train OOF) T = {T_final:.4f}")

    # ---------------- final fit ----------------
    final, t_final, final_info = fit_ranker(tr, cv_trips)
    print(f"final fit on {len(cv_trips)} trips in {t_final:.1f}s")
    mx = tr.rows(excluded)                                               # never trained on: out-of-sample
    if mx.any():
        oof[mx] = final.predict(tr.X[mx])
        proba_tr[mx] = group_softmax(oof[mx], tr.gid[mx], T_final)

    t0 = time.perf_counter()
    s_st = final.predict(st.X)
    p_st = group_softmax(s_st, st.gid, T_final)
    t_inf = time.perf_counter() - t0
    st_trips = sorted(np.unique(st.trip))
    stm = candidate_metrics(st, st.labelled, s_st, p_st)
    stm["group_nll"] = group_nll(s_st[st.labelled], st.y[st.labelled], st.gid[st.labelled], T_final)

    write_scores(tr, oof, proba_tr, sorted(np.unique(tr.trip)))
    write_scores(st, s_st, p_st, st_trips)

    # ---------------- artifacts ----------------
    mpath = os.path.join(ARTIFACT_DIR, f"{MODEL}.txt")
    final.booster_.save_model(mpath)
    booster = final.booster_
    n_trees = booster.num_trees()
    n_leaves = int(sum(t["num_leaves"] for t in booster.dump_model()["tree_info"]))
    size_mb = os.path.getsize(mpath) / 1e6
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"model": MODEL, "features": EMISSION_FEATURES, "temperature": T_final, "config": CONFIG,
            "seed": SEED, "fold_seed": F.get("seed"), "trained_on_trips": cv_trips,
            "excluded_trips": excluded, "library_versions": versions(), "training_date": now,
            "probability": "p = softmax(score / temperature) within each GPS point's candidate group"}
    with open(os.path.join(ARTIFACT_DIR, f"{MODEL}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    # ---------------- plots ----------------
    lab = st.labelled
    plot_confusion(st.y[lab], group_top1(s_st[lab], st.gid[lab]).astype(int),
                   "student_test - Top-1 rule", os.path.join(REPORT_DIR, "confusion_matrix_top1.png"))
    plot_confusion(st.y[lab], (p_st[lab] >= 0.5).astype(int),
                   "student_test - threshold rule (p >= 0.5)", os.path.join(REPORT_DIR, "confusion_matrix_thr05.png"))
    plot_calibration(st.y[lab], p_st[lab], os.path.join(REPORT_DIR, "calibration.png"))
    plot_importance(booster, os.path.join(REPORT_DIR, "feature_importance_gain.png"))

    # ---------------- route level: classifier-only vs classifier+HMM ----------------
    from src.hmm.run_decoder import run as run_decoder
    route_st = run_decoder(MODEL, "student_test", PROCESSED_DIR, FEATURES_DIR, "data/predictions", "reports/hmm", SCORES_DIR)
    route_tr = run_decoder(MODEL, "train", PROCESSED_DIR, FEATURES_DIR, "data/predictions", "reports/hmm", SCORES_DIR)
    route_tr = route_tr[route_tr.trip.astype(str).str.zfill(8).isin(cv_trips)]   # out-of-fold trips only

    # ---------------- report ----------------
    timings = {"train_time_per_fold_s": [round(f["train_time_s"], 2) for f in fold_info],
               "final_fit_s": round(t_final, 2), "inference_total_s": round(t_inf, 4),
               "inference_us_per_row": t_inf / len(st.y) * 1e6, "inference_ms_per_trip": t_inf / len(st_trips) * 1e3,
               "student_test_rows_scored": int(len(st.y))}
    complexity = {"n_trees": n_trees, "total_leaves": n_leaves, "model_file_mb": size_mb}
    dropped = {"cv_folds": [f["dropped_groups_no_true"] for f in fold_info],
               "final_fit": final_info["dropped_groups_no_true"]}
    md = report_md(cv, stm, route_st, route_tr, T_final, timings, complexity, dropped, final_info, excluded, F)
    with open(os.path.join(REPORT_DIR, "metrics.md"), "w") as fh:
        fh.write(md)

    log = {"model": MODEL, "family": FAMILY, "timestamp": now, "config": CONFIG, "seeds": {
               "model_random_state": SEED, "numpy": SEED, "shared_fold_seed": F.get("seed")},
           "features": EMISSION_FEATURES, "folds_file": FOLDS_PATH, "cv_trips": len(cv_trips),
           "excluded_trips": excluded,
           "rows": {"train_all": int(len(tr.y)), "train_labelled": int(tr.labelled.sum()),
                    "student_test_all": int(len(st.y)), "student_test_labelled": int(st.labelled.sum())},
           "groups": {"train_labelled": int(len(np.unique(tr.gid[tr.labelled]))),
                      "student_test_labelled": int(len(np.unique(st.gid[st.labelled]))),
                      "final_fit_train_groups": final_info["train_groups"]},
           "dropped_training_groups_no_true_arc": dropped, "temperature": T_final,
           "temperature_per_fold_for_cv": cv.temperature.round(4).tolist(),
           "timings": timings, "complexity": complexity, "library_versions": versions()}
    with open(os.path.join(REPORT_DIR, "experiment_log.json"), "w") as fh:
        json.dump(log, fh, indent=1, default=float)

    append_comparison(cv, stm, route_st, T_final, t_final, timings, size_mb)
    print("\n" + md)


def _route_means(df):
    return df.groupby("method", sort=False).mean(numeric_only=True)


def report_md(cv, stm, route_st, route_tr, T, timings, cx, dropped, final_info, excluded, F) -> str:
    def ms(c, pct=True):
        if pct:
            return f"{cv[c].mean() * 100:.2f} ± {cv[c].std() * 100:.2f}"
        return f"{cv[c].mean():.4f} ± {cv[c].std():.4f}"

    def v(x, pct=True):
        return f"{x * 100:.2f}" if pct else f"{x:.4f}"

    rows = [("Observation Top-1 accuracy (%)", "obs_top1", True), ("Candidate miss rate (%)", "candidate_miss_rate", True),
            ("Log loss (candidate, binary)", "logloss", False), ("Softmax NLL per point (multiclass)", "group_nll", False),
            ("ROC-AUC", "roc_auc", False)]
    L = ["# Model 5 - LightGBM LambdaRank candidate ranker\n",
         "Family: **" + FAMILY + "**. Untuned configuration (tuning comes in Section 7): "
         "`n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, "
         "colsample_bytree=0.8, random_state=42`, no early stopping. Features: the 17 `EMISSION_FEATURES`.\n",
         f"Folds: shared team folds `outputs/splits/folds.json` (5 trip-grouped folds, seed {F.get('seed')}), "
         f"{sum(len(f) for f in F['folds'])} train trips. Trips {', '.join(excluded)} are listed there as "
         "label-unreliable and are excluded from training and CV; they are scored out-of-sample by the final "
         "model. The final model is trained on the same 78 trips. student_test (10 trips) is used for "
         "evaluation only.\n",
         "## From ranking scores to probabilities\n",
         "LambdaRank outputs an unbounded relevance score that is only meaningful *within* one GPS point's "
         "candidate group, so it cannot be read as a probability directly. Because exactly one candidate per "
         "point is the true arc, we turn the scores of each group into a distribution with a softmax, "
         "p = softmax(score / T), which sums to 1 over the point's candidates and keeps the ranking unchanged. "
         f"The temperature T sets how peaked the distribution is; it is fitted on train out-of-fold scores only "
         f"by minimising the multiclass log loss (bounded scalar search in [0.05, 20]), giving **T = {T:.4f}**. "
         "For the CV estimates each fold uses a temperature fitted on the other four folds, so no fold's "
         "probabilities are calibrated on itself; student_test never influences T.\n",
         "## Candidate and observation level\n",
         "| Metric | CV (mean ± std over 5 folds) | student_test |", "|---|---:|---:|"]
    for name, c, pct in rows:
        L.append(f"| {name} | {ms(c, pct)} | {v(stm[c], pct)} |")
    L += ["", "Two decision rules on the candidate rows (positive class = true arc; values in %):\n",
          "| Rule | Metric | CV (mean ± std) | student_test |", "|---|---|---:|---:|"]
    for rule, label in (("top1", "a) Top-1 (main rule)"), ("thr05", "b) Threshold p ≥ 0.5")):
        for mname in ("accuracy", "precision", "recall", "f1"):
            L.append(f"| {label} | {mname.capitalize()} | {ms(f'{rule}_{mname}')} | {v(stm[f'{rule}_{mname}'])} |")
    L += ["", "Confusion matrices (student_test): `confusion_matrix_top1.png`, `confusion_matrix_thr05.png`. "
              "Reliability plot: `calibration.png`. Feature importance: `feature_importance_gain.png`. "
              "Per-fold numbers: `cv_folds.csv`.\n",
          "Observation Top-1 counts every labelled GPS point, including points with no candidates or whose true "
          "arc is not a candidate (these are always wrong), so it is capped at 100 − candidate miss rate.\n"]

    L += ["## Route level: classifier-only vs classifier + HMM\n",
          "Decoder: `src.hmm.run_decoder` with this model's probabilities as emissions "
          "(emission_logp = log(clip(p, 1e-9, 1))) and the same Newson & Krumm transition "
          "(beta estimated on train) as the Gaussian run. Train = out-of-fold scores on the 78 CV trips. "
          "Values are means over trips, in %.\n"]
    cols = [("top1_acc", "Top-1"), ("continuity_pct", "Continuity"), ("dir_precision", "Dir P"),
            ("dir_recall", "Dir R"), ("dir_f1", "Dir F1"), ("undir_precision", "Undir P"), ("undir_recall", "Undir R"),
            ("undir_f1", "Undir F1"), ("dir_len_precision", "Dir len-P"), ("dir_len_recall", "Dir len-R"),
            ("route_gaps", "Route gaps"), ("n_chain_breaks", "Chain breaks")]
    gauss_p = os.path.join("reports", "hmm", "gaussian_student_test_metrics.csv")
    gauss = _route_means(pd.read_csv(gauss_p)) if os.path.exists(gauss_p) else None
    L += ["| Partition | Method | " + " | ".join(n for _, n in cols) + " |", "|---|---|" + "---:|" * len(cols)]
    for part, df in (("student_test", route_st), ("train (OOF)", route_tr)):
        mm = _route_means(df)
        for meth in ("classifier", "classifier+hmm"):
            label = "LambdaRank only" if meth == "classifier" else "LambdaRank + HMM"
            L.append(f"| {part} | {label} | " + " | ".join(
                "–" if pd.isna(mm.loc[meth, c]) else f"{mm.loc[meth, c]:.2f}" for c, _ in cols) + " |")
    if gauss is not None:
        L.append("| student_test | *Gaussian + HMM (reference)* | " + " | ".join(
            "–" if pd.isna(gauss.loc["classifier+hmm", c]) else f"{gauss.loc['classifier+hmm', c]:.2f}"
            for c, _ in cols) + " |")

    L += ["", "## Cost\n",
          "| Item | Value |", "|---|---:|",
          *[f"| Training time, fold {i} (s) | {t:.2f} |" for i, t in enumerate(timings["train_time_per_fold_s"])],
          f"| Training time, final fit (s) | {timings['final_fit_s']:.2f} |",
          f"| Inference, student_test total (s) | {timings['inference_total_s']:.3f} |",
          f"| Inference per row (µs) | {timings['inference_us_per_row']:.2f} |",
          f"| Inference per trip (ms) | {timings['inference_ms_per_trip']:.1f} |",
          f"| Trees | {cx['n_trees']} |", f"| Total leaves | {cx['total_leaves']:,} |",
          f"| Model file size (MB) | {cx['model_file_mb']:.2f} |", "",
          "Inference time covers LightGBM scoring plus the group softmax for all "
          f"{timings['student_test_rows_scored']:,} student_test candidate rows (feature building and decoding excluded).\n",
          "## Training data notes\n",
          f"Points whose true arc is not among their candidates carry no ranking signal and were dropped from "
          f"training only: {dropped['cv_folds']} groups per CV fold, {dropped['final_fit']} in the final fit "
          f"(final fit trained on {final_info['train_groups']:,} groups / {final_info['train_rows']:,} rows). "
          "They remain in every evaluation, where they count as wrong.\n"]
    return "\n".join(L)


COMPARISON_COLS = ["model", "family", "key_config", "cv_top1_mean", "cv_top1_std", "cv_logloss", "st_top1",
                   "st_accuracy", "st_precision", "st_recall", "st_f1", "st_logloss", "st_continuity_classifier",
                   "st_continuity_hmm", "st_dir_f1_hmm", "train_time_s", "infer_us_per_row", "model_size_mb"]


def append_comparison(cv, stm, route_st, T, t_final, timings, size_mb):
    path = os.path.join("reports", "models", "comparison.csv")
    mm = _route_means(route_st)
    row = {"model": MODEL, "family": FAMILY,
           "key_config": f"lambdarank; 500 trees; lr 0.05; 31 leaves; min_child 50; subsample 0.8; colsample 0.8; "
                         f"softmax T={T:.3f}; no tuning",
           "cv_top1_mean": cv.obs_top1.mean(), "cv_top1_std": cv.obs_top1.std(), "cv_logloss": cv.logloss.mean(),
           "st_top1": stm["obs_top1"], "st_accuracy": stm["top1_accuracy"], "st_precision": stm["top1_precision"],
           "st_recall": stm["top1_recall"], "st_f1": stm["top1_f1"], "st_logloss": stm["logloss"],
           "st_continuity_classifier": mm.loc["classifier", "continuity_pct"] / 100,
           "st_continuity_hmm": mm.loc["classifier+hmm", "continuity_pct"] / 100,
           "st_dir_f1_hmm": mm.loc["classifier+hmm", "dir_f1"] / 100,
           "train_time_s": t_final, "infer_us_per_row": timings["inference_us_per_row"], "model_size_mb": size_mb}
    if os.path.exists(path):
        old = pd.read_csv(path)
        old = old[old.model != MODEL]                       # re-runs replace this model's row
        out = pd.concat([old, pd.DataFrame([row])[old.columns]], ignore_index=True)
    else:
        out = pd.DataFrame([row])[COMPARISON_COLS]
    out.to_csv(path, index=False, float_format="%.6g")


if __name__ == "__main__":
    main()
