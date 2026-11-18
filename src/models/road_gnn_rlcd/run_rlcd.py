"""RLCD ablation on a frozen Model 2 scorer (see pathsnap/rlcd.py for the method).

Compared on identical frozen folds (policies are cross-fitted: the policy for
fold k is trained on OOF predictions of the other folds):
  1 fixed_raw    : single best window (chosen on CV OOF Brier), no calibration
  2 fixed_temp   : same window, per-window temperature scaling (fit on OOF)
  3 direct       : full-information selector (regress the three Briers, act argmin)
  3b expected    : full-information exact expected reward  max sum_a pi(a|s) r_a
  4 rlcd         : REINFORCE (sampled action, bandit feedback) + value baseline
  (oracle)       : per-fix best window using the label -- an UPPER BOUND only
Student-test: policies trained on all 78 training trips' OOF predictions, applied
to the final scorer's student-test outputs.

python -m src.models.road_gnn_rlcd.run_rlcd [mp|nomp]  -> results/rlcd_<variant>/ (this folder)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pathsnap.context import prepare  # noqa: E402
from pathsnap.evaluate import evaluate_many, summarise  # noqa: E402
from pathsnap.io import list_trips, load_trip  # noqa: E402
from src.models.road_gnn_rlcd.model import WINDOWS_S  # noqa: E402
from src.models.road_gnn_rlcd.rlcd import ACTIONS, STATE_NAMES, brier_per_fix, state_features, train_policy  # noqa: E402
from pathsnap.splits import SEED, load_folds  # noqa: E402

variant = sys.argv[1] if len(sys.argv) > 1 else "mp"
PRED = os.path.join(ROOT, "outputs", "preds", f"m2_{variant}")
RES = os.path.join(HERE, "results", f"rlcd_{variant}")
os.makedirs(RES, exist_ok=True)
temps = json.load(open(os.path.join(HERE, "results", f"m2_{variant}", "temperature.json")))
TW = [temps[str(w)]["T"] for w in range(ACTIONS)]


def flat_softmax(z, ptr, T):
    out = np.zeros(len(z))
    for f in range(len(ptr) - 1):
        s, e = ptr[f], ptr[f + 1]
        if e > s:
            a = z[s:e] / T
            a = np.exp(a - a.max())
            out[s:e] = a / a.sum()
    return out


def trip_data(part, tid, prefix):
    ctx = prepare(load_trip(part, tid))
    c = ctx.cands
    Z = np.load(os.path.join(PRED, f"{prefix}_{tid}.npz"))["logits"]
    L = np.load(os.path.join(ROOT, "outputs", "labels", part, f"{tid}.npz"))
    y = np.full(ctx.T, -1)
    hit = np.nonzero(c.arc == L["label"][c.obs])[0]
    y[c.obs[hit]] = hit
    raw = [flat_softmax(Z[w], c.ptr, 1.0) for w in range(ACTIONS)]
    cal = [flat_softmax(Z[w], c.ptr, TW[w]) for w in range(ACTIONS)]
    R_raw = -np.stack([brier_per_fix(p, c.ptr, c.obs, y) for p in raw], 1)
    R = -np.stack([brier_per_fix(p, c.ptr, c.obs, y) for p in cal], 1)
    S = state_features(ctx.t, ctx.xy, c.ptr, c.dist, cal)
    return dict(tid=tid, part=part, ptr=c.ptr, obs=c.obs, y=y, far=L["far"], amb=L["ambiguous"],
                raw=raw, cal=cal, R=R, R_raw=R_raw, S=S)


def compose(d, actions, calibrated=True):
    P = d["cal"] if calibrated else d["raw"]
    a_row = actions[d["obs"]]
    return np.choose(a_row, P)


def reliability(items, bins=10):
    conf, corr = [], []
    for d, p in items:
        ptr, y = d["ptr"], d["y"]
        for f in range(len(ptr) - 1):
            s, e = ptr[f], ptr[f + 1]
            if e > s:
                k = s + int(np.argmax(p[s:e]))
                conf.append(p[k]); corr.append(k == y[f])
    conf, corr = np.array(conf), np.array(corr, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, bins - 1)
    rows = []
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(dict(bin_lo=edges[b], bin_hi=edges[b + 1], count=int(m.sum()),
                             mean_conf=float(conf[m].mean()), accuracy=float(corr[m].mean())))
            ece += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return rows, float(ece)


t0 = time.time()
folds = load_folds()["folds"]
fold_of = {t: k for k, f in enumerate(folds) for t in f}
tr_data = {t: trip_data("train", t, "oof") for t in sorted(fold_of)}
st_data = {t: trip_data("student_test", t, "st") for t in list_trips("student_test")}
print("data", round(time.time() - t0, 1), "s", flush=True)

# fixed window chosen on CV OOF (training data only)
cvR_raw = np.concatenate([d["R_raw"] for d in tr_data.values()]).mean(0)
cvR = np.concatenate([d["R"] for d in tr_data.values()]).mean(0)
w_raw, w_cal = int(np.argmax(cvR_raw)), int(np.argmax(cvR))
print("CV mean reward per window raw", cvR_raw.round(4), "cal", cvR.round(4))


def stack(ds, train=True):
    S = np.concatenate([d["S"] for d in ds])
    R = np.concatenate([d["R"] for d in ds])
    keep = np.concatenate([~d["far"] for d in ds]) if train else np.ones(len(S), bool)
    return S[keep], R[keep]


modes = ["direct", "expected", "rlcd"]
actions = {m: {} for m in ["fixed_raw", "fixed_temp", "oracle"] + modes}
logs = []
for name, data in [("cv", tr_data), ("st", st_data)]:
    for t, d in data.items():
        T = len(d["ptr"]) - 1
        actions["fixed_raw"][(name, t)] = np.full(T, w_raw)
        actions["fixed_temp"][(name, t)] = np.full(T, w_cal)
        actions["oracle"][(name, t)] = np.argmax(d["R"], 1)
for m in modes:
    ts = time.time()
    for k in range(len(folds)):   # cross-fitted CV
        S, R = stack([tr_data[t] for t in tr_data if fold_of[t] != k])
        act, log = train_policy(S, R, m, seed=SEED + k)
        logs += [dict(mode=m, fold=k, **r) for r in log]
        for t in folds[k]:
            actions[m][("cv", t)] = act(tr_data[t]["S"])
    S, R = stack(list(tr_data.values()))
    act, log = train_policy(S, R, m, seed=SEED)
    logs += [dict(mode=m, fold="all", **r) for r in log]
    for t, d in st_data.items():
        actions[m][("st", t)] = act(d["S"])
    print(m, "trained", round(time.time() - ts, 1), "s", flush=True)
pd.DataFrame(logs).to_csv(os.path.join(RES, "policy_training_log.csv"), index=False)

rows, rel_rows, hmm_rows = [], [], []
for m, acts in actions.items():
    for name, data, part in [("cv", tr_data, "train"), ("st", st_data, "student_test")]:
        items = []
        allR, allA, nf, ll, top = [], [], 0, 0.0, 0
        for t, d in data.items():
            a = acts[(name, t)]
            p = compose(d, a, calibrated=(m != "fixed_raw"))
            items.append((d, p))
            Rsrc = d["R_raw"] if m == "fixed_raw" else d["R"]
            allR.append(Rsrc[np.arange(len(a)), a]); allA.append(a)
            y = d["y"]
            pl = np.where(y >= 0, p[np.maximum(y, 0)], 0)
            ll += -np.log(np.clip(pl, 1e-6, 1)).sum()
            nf += len(y)
            for f in range(len(y)):
                s, e = d["ptr"][f], d["ptr"][f + 1]
                if e > s and y[f] >= 0:
                    top += (s + int(np.argmax(p[s:e]))) == y[f]
        rel, ece = reliability(items)
        A = np.concatenate(allA)
        rows.append(dict(method=m, split=name, fixes=nf, brier=-np.concatenate(allR).mean(), log_loss=ll / nf,
                         top1=top / nf, ece=ece, **{f"frac_a{w}": float((A == w).mean()) for w in range(ACTIONS)}))
        rel_rows += [dict(method=m, split=name, **r) for r in rel]
        if m != "oracle":
            hmm_rows.append((m, name, part, [(part, d["tid"], p) for d, p in items]))
res = pd.DataFrame(rows)
res.to_csv(os.path.join(RES, "classifier_only.csv"), index=False)
pd.DataFrame(rel_rows).to_csv(os.path.join(RES, "reliability.csv"), index=False)
pd.set_option("display.width", 250)
print(res.round(4).to_string(index=False), flush=True)

summ = []
for m, name, part, items in hmm_rows:
    df = pd.DataFrame(evaluate_many(items))
    df.to_csv(os.path.join(RES, f"per_trip_{m}_{name}.csv"), index=False)
    s = summarise(df)
    s.insert(0, "split", name)
    s.insert(0, "method", m)
    summ.append(s)
summ = pd.concat(summ)
summ.to_csv(os.path.join(RES, "summary_hmm.csv"), index=False)
json.dump(dict(variant=variant, windows_s=WINDOWS_S, fixed_window_raw=WINDOWS_S[w_raw],
               fixed_window_cal=WINDOWS_S[w_cal], temperatures=TW, state_features=STATE_NAMES,
               wall_s=time.time() - t0), open(os.path.join(RES, "run_info.json"), "w"), indent=1)
print(summ[["method", "split", "mode", "top1", "macro_f1_tripmean", "brier", "log_loss", "continuity", "route_f1"]]
      .round(4).to_string(index=False))
