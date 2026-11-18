"""Train + evaluate Model 1 (variant C: raw + IMM features, listwise MLP).

Steps
 1. 5-fold trip-grouped CV (frozen folds) -> out-of-fold (OOF) logits, per-epoch train/val loss.
 2. Fit one temperature on the OOF logits (training trips only).
 3. Refit on all 78 label-reliable training trips; apply the OOF temperature.
 4. Predict the 10 student-test trips (development comparison).
 5. Evaluate classifier-only and classifier+HMM for CV (OOF) and student-test.
Outputs: artifacts/imm_1hz_listwise.pt, results/imm_1hz/*.csv (this folder), outputs/preds/m1/...
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ROOT)
from pathsnap.evaluate import evaluate_many, summarise  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.listwise import ListwiseScorer, fit_temperature, pad, unpad  # noqa: E402
from src.models.imm_2ekf.model1 import FEATURE_NAMES  # noqa: E402
from pathsnap.splits import SEED, load_folds  # noqa: E402

CACHE = os.path.join(ROOT, "outputs", "cache", "m1")
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "imm_1hz")
PRED = os.path.join(ROOT, "outputs", "preds", "m1")
for d in (RES, PRED, os.path.join(HERE, "artifacts")):
    os.makedirs(d, exist_ok=True)
torch.set_num_threads(8)


def load(part, tids, train=False):
    Xs, Ms, ys, meta = [], [], [], []
    for t in tids:
        z = np.load(os.path.join(CACHE, part, f"{t}.npz"))
        X, M, y = pad(z["F"], z["ptr"], z["y_row"])
        if train:
            y = np.where(z["far"], -1, y)   # loss excludes off-road / missing labels
        Xs.append(X); Ms.append(M); ys.append(y); meta.append((t, z["ptr"], len(X)))
    return np.concatenate(Xs), np.concatenate(Ms), np.concatenate(ys), meta


def split_back(P, meta):
    out, i = {}, 0
    for t, ptr, n in meta:
        out[t] = unpad(P[i:i + n], ptr)
        i += n
    return out


folds = load_folds()["folds"]
train_all = sorted(sum(folds, []))
oof_logits, oof_meta, hist = {}, {}, []
t_cv = time.time()
for k in range(len(folds)):
    tr = [t for j, f in enumerate(folds) if j != k for t in f]
    X, M, y, _ = load("train", tr, train=True)
    Xv, Mv, yv, mv = load("train", folds[k])
    _, _, yv_loss, _ = load("train", folds[k], train=True)
    s = ListwiseScorer(X.shape[2], seed=SEED + k).fit(X, M, y, Xv, Mv, yv_loss)
    Z = s.logits(Xv, Mv)
    for t, ptr, n in mv:
        pass
    i = 0
    for t, ptr, n in mv:
        oof_logits[t] = (Z[i:i + n], Mv[i:i + n], yv_loss[i:i + n], ptr)
        i += n
    for h in s.history:
        hist.append(dict(fold=k, **h))
    print(f"fold {k}: last train {s.history[-1]['train_loss']:.4f} val {s.history[-1]['val_loss']:.4f} "
          f"({s.train_time:.1f}s)", flush=True)
cv_time = time.time() - t_cv
pd.DataFrame(hist).to_csv(os.path.join(RES, "cv_history.csv"), index=False)

# temperature on OOF
Zc = np.concatenate([v[0] for v in oof_logits.values()])
Mc = np.concatenate([v[1] for v in oof_logits.values()])
yc = np.concatenate([v[2] for v in oof_logits.values()])
T, nll_T, nll_1 = fit_temperature(Zc, Mc, yc)
print(f"temperature {T:.3f}: OOF NLL {nll_1:.4f} -> {nll_T:.4f}")

# refit on all training trips
X, M, y, _ = load("train", train_all, train=True)
final = ListwiseScorer(X.shape[2], seed=SEED).fit(X, M, y)
final.temperature = T
torch.save(dict(state=final.state(), feature_names=FEATURE_NAMES, kind="model1_imm_listwise_mlp",
                tracker_config=json.load(open(os.path.join(HERE, "configs", "tracker_1hz.json")))["params"]),
           os.path.join(HERE, "artifacts", "imm_1hz_listwise.pt"))
print(f"final fit {final.train_time:.1f}s")

# predictions
helper = ListwiseScorer(X.shape[2])
items_cv = []
for t, (Z, Mk, _, ptr) in oof_logits.items():
    P = helper.proba_from_logits(Z, Mk, T)
    p = unpad(P, ptr)
    np.save(os.path.join(PRED, f"oof_{t}.npy"), p)
    items_cv.append(("train", t, p))
st = list_trips("student_test")
Xs, Ms, _, ms = load("student_test", st)
Ps = split_back(final.predict_proba_padded(Xs, Ms), ms)
items_st = []
for t, p in Ps.items():
    np.save(os.path.join(PRED, f"st_{t}.npy"), p)
    items_st.append(("student_test", t, p))

cv = pd.DataFrame(evaluate_many(items_cv))
stt = pd.DataFrame(evaluate_many(items_st))
cv.to_csv(os.path.join(RES, "cv_per_trip.csv"), index=False)
stt.to_csv(os.path.join(RES, "student_test_per_trip.csv"), index=False)
s_cv, s_st = summarise(cv), summarise(stt)
s_cv.insert(0, "split", "cv_oof")
s_st.insert(0, "split", "student_test")
summ = pd.concat([s_cv, s_st])
summ.to_csv(os.path.join(RES, "summary.csv"), index=False)
json.dump(dict(temperature=T, oof_nll_T1=nll_1, oof_nll_T=nll_T, cv_train_wall_s=cv_time,
               final_train_s=final.train_time, n_features=len(FEATURE_NAMES), seed=SEED),
          open(os.path.join(RES, "run_info.json"), "w"), indent=1)
pd.set_option("display.width", 250)
print(summ.round(4).to_string(index=False))
