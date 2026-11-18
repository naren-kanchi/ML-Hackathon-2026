"""Report figures, generated only from committed experiment results.

python scripts/make_figures.py -> reports/figures/*.png
  01_cv_top1_models.png          CV Top-1 per model, classifier vs +Viterbi (fold sd error bars)
  02_cv_macro_f1_models.png      CV trip-mean macro F1 per model, classifier vs +Viterbi
  03_student_test_models.png     student-test Top-1 and macro F1 per model (+Viterbi)
  04_bayes_opt_lgbm.png          Bayesian optimisation trial history (stage A ranker, stage B decoder)
  05_tuned_vs_untuned_lgbm.png   LightGBM untuned vs tuned
  06_training_curves.png         epoch-wise train/validation loss (models with epoch histories only)
Palette: validated categorical slots 1-2 of the dataviz reference palette.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "figures")
S1, S2 = "#2a78d6", "#eb6834"          # series 1 (classifier / train), series 2 (+Viterbi / validation)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
MODELS = [("lgbm_ranker", "LightGBM LambdaRank"), ("team_plus_imm", "IMM + 2 EKF"), ("gnn_mp", "Road-graph GNN"),
          ("logistic_regression", "Logistic Regression"), ("histgradientboosting", "HistGradientBoosting")]


def style(ax, xlabel=None, ylabel=None):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def paired_bars(ax, labels, a, b, a_err=None, b_err=None, names=("Classifier only", "+ HMM/Viterbi"), xlim=None,
                legend=True):
    y = np.arange(len(labels))
    h = 0.36
    ax.barh(y - h / 2 - 0.01, a, h, color=S1, xerr=a_err, error_kw=dict(ecolor=INK2, lw=1, capsize=2), label=names[0])
    ax.barh(y + h / 2 + 0.01, b, h, color=S2, xerr=b_err, error_kw=dict(ecolor=INK2, lw=1, capsize=2), label=names[1])
    ea = np.zeros(len(a)) if a_err is None else np.asarray(a_err)
    eb = np.zeros(len(b)) if b_err is None else np.asarray(b_err)
    pad = 0.008 * ((xlim[1] - xlim[0]) if xlim else 1.0)
    for yi, v, e in zip(y - h / 2, a, ea):      # label sits past the error bar
        ax.text(v + e + pad, yi, f"{v:.3f}", va="center", fontsize=8, color=INK2)
    for yi, v, e in zip(y + h / 2, b, eb):
        ax.text(v + e + pad, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    if xlim:
        ax.set_xlim(*xlim)
    if legend:
        ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK2)


def fig_models():
    s = pd.read_csv(os.path.join(ROOT, "experiments", "results", "common_eval", "summary.csv"))
    get = lambda m, sp, mode, c: float(s[(s.model == m) & (s.split == sp) & (s["mode"] == mode)][c].iloc[0])  # noqa: E731
    lab = [n for _, n in MODELS]
    note = ("Same points (team-labelled), same labels, same decoder; CV = out-of-fold on 78 training trips "
            "(5 frozen folds); error bars = fold sd")
    for fname, col, title, sd in (("01_cv_top1_models.png", "top1", "Cross-validation Top-1 accuracy", True),
                                  ("02_cv_macro_f1_models.png", "macro_f1_tripmean",
                                   "Cross-validation macro F1 (per-trip arc classes, mean over trips)", False)):
        fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF)
        a = [get(m, "cv_oof", "clf", col) for m, _ in MODELS]
        b = [get(m, "cv_oof", "hmm", col) for m, _ in MODELS]
        ea = [get(m, "cv_oof", "clf", "fold_top1_sd") for m, _ in MODELS] if sd else None
        eb = [get(m, "cv_oof", "hmm", "fold_top1_sd") for m, _ in MODELS] if sd else None
        paired_bars(ax, lab, a, b, ea, eb, xlim=(0.5, 1.06))
        style(ax, xlabel=col.replace("_tripmean", "").replace("top1", "Top-1").replace("macro_f1", "macro F1"))
        ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
        fig.text(0.01, 0.01, note, fontsize=7, color=MUTED)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(os.path.join(OUT, fname), dpi=160)
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.3), facecolor=SURF)
    for i, (ax, col, t) in enumerate(zip(axs, ("top1", "macro_f1_tripmean"), ("Top-1 accuracy", "macro F1"))):
        a = [get(m, "student_test", "clf", col) for m, _ in MODELS]
        b = [get(m, "student_test", "hmm", col) for m, _ in MODELS]
        paired_bars(ax, lab if i == 0 else [""] * len(lab), a, b, xlim=(0.4, 1.06), legend=False)
        style(ax)
        ax.set_title(f"Student-test {t}", loc="left", color=INK, fontsize=11, fontweight="bold")
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper right", ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    fig.text(0.01, 0.01, "10 student-test trips, 13,340 team-labelled points; models refit on the 78 CV trips",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(os.path.join(OUT, "03_student_test_models.png"), dpi=160)
    plt.close(fig)


def fig_bayes():
    d = os.path.join(ROOT, "experiments", "results", "bayes_opt_lgbm")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURF)
    for ax, f, t in zip(axs, ("stageA_trials.csv", "stageB_trials.csv"),
                        ("Stage A: ranker (CV Top-1, classifier)",
                         "Stage B: decoder (CV Top-1 after Viterbi)")):
        tr = pd.read_csv(os.path.join(d, f))
        ax.plot(tr.number, tr.value, "o", color=S1, ms=6, mec=SURF, mew=1.5, label="Trial")
        ax.plot(tr.number, tr.value.cummax(), "-", color=S2, lw=2, drawstyle="steps-post", label="Best so far")
        ax.axhline(tr.value.iloc[0], color=MUTED, lw=1, ls="--")
        ax.text(tr.number.max(), tr.value.iloc[0], " untuned (trial 0)", va="bottom", ha="right", fontsize=7,
                color=MUTED)
        style(ax, xlabel="Trial", ylabel="Mean fold Top-1")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_title(t, loc="left", color=INK, fontsize=9, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.text(0.01, 0.01, "Optuna TPE, seed 20261118; 5 frozen trip-grouped folds of the 78 training trips",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(os.path.join(OUT, "04_bayes_opt_lgbm.png"), dpi=160)
    plt.close(fig)
    t = pd.read_csv(os.path.join(d, "tuned_vs_untuned.csv"))
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURF)
    labels = ["CV Top-1 (classifier)", "CV Top-1 (+Viterbi)", "Student-test Top-1 (+Viterbi)",
              "Student-test macro F1 (+Viterbi)"]
    g = lambda n, sp, c: float(t[(t.model == n) & (t.split == sp)][c].iloc[0])  # noqa: E731
    vals = {n: [g(n, "cv_5fold_oof", "top1_clf"), g(n, "cv_5fold_oof", "top1_hmm"), g(n, "student_test", "top1_hmm"),
                g(n, "student_test", "macro_f1_hmm")] for n in ("untuned", "tuned")}
    paired_bars(ax, labels, vals["untuned"], vals["tuned"], names=("Untuned", "Tuned (Bayesian optimisation)"),
                xlim=(0.85, 1.0), legend=False)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    style(ax)
    ax.set_title("LightGBM LambdaRank: untuned vs tuned", loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "05_tuned_vs_untuned_lgbm.png"), dpi=160)
    plt.close(fig)


def fig_curves():
    imm = pd.read_csv(os.path.join(ROOT, "src", "models", "imm_2ekf", "results", "imm_teamdata", "cv_history.csv"))
    imm = imm[imm.variant == "team_plus_imm"].groupby("epoch")[["train_loss", "val_loss"]].mean()
    gnn = pd.read_csv(os.path.join(ROOT, "src", "models", "road_gnn_rlcd", "results", "m2_mp", "cv_history.csv"))
    gnn = gnn.groupby("epoch")[["train_loss", "val_loss_w2"]].mean().rename(columns={"val_loss_w2": "val_loss"})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=SURF)
    for ax, df, t in zip(axs, (imm, gnn), ("IMM + 2 EKF (listwise MLP)",
                                           "Road-graph GNN (message passing; val = 15 s window)")):
        ax.plot(df.index, df.train_loss, color=S1, lw=2, label="Training loss")
        ax.plot(df.index, df.val_loss, color=S2, lw=2, label="Validation loss")
        style(ax, xlabel="Epoch", ylabel="Cross-entropy (mean of 5 folds)")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_title(t, loc="left", color=INK, fontsize=10, fontweight="bold")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.text(0.01, 0.01, "Only models with epoch-wise training produce curves; tree models and logistic regression "
             "have none.  GNN training loss averages all three windows.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(OUT, "06_training_curves.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_models()
    fig_bayes()
    fig_curves()
    print(sorted(f for f in os.listdir(OUT) if f.endswith(".png")))
