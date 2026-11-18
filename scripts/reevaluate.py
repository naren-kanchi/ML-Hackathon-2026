"""Recompute every decoder/route evaluation from SAVED predictions (no retraining).

Use after changing pathsnap/hmm.py or pathsnap/metrics.py so the committed
result CSVs match the code. Scorer outputs are unchanged, so Top-1, macro F1,
log loss and Brier must come out identical; only continuity / route metrics move.

python scripts/reevaluate.py   (then: python -m src.models.road_gnn_rlcd.run_rlcd mp)
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pathsnap.context import prepare  # noqa: E402
from pathsnap.evaluate import evaluate_many, summarise  # noqa: E402
from pathsnap.io import load_trip  # noqa: E402


def write(res_dir, items_cv, items_st, extra=None):
    cv, st = pd.DataFrame(evaluate_many(items_cv)), pd.DataFrame(evaluate_many(items_st))
    cv.to_csv(os.path.join(res_dir, "cv_per_trip.csv"), index=False)
    st.to_csv(os.path.join(res_dir, "student_test_per_trip.csv"), index=False)
    a, b = summarise(cv), summarise(st)
    a.insert(0, "split", "cv_oof")
    b.insert(0, "split", "student_test")
    s = pd.concat([a, b])
    for k, v in (extra or {}).items():
        s.insert(1, k, v)
    s.to_csv(os.path.join(res_dir, "summary.csv"), index=False)
    print(res_dir)
    print(s[["split", "mode", "top1", "macro_f1_tripmean", "continuity", "route_f1", "route_f1_trim",
             "route_lcs"]].round(4).to_string(index=False), flush=True)


def flat_softmax(z, ptr, T):
    out = np.zeros(len(z))
    for f in range(len(ptr) - 1):
        s, e = ptr[f], ptr[f + 1]
        if e > s:
            a = z[s:e] / T
            a = np.exp(a - a.max())
            out[s:e] = a / a.sum()
    return out


# Model 1 (IMM, 1 Hz)
P = os.path.join(ROOT, "outputs", "preds", "m1")
items = {k: [("train" if k == "oof" else "student_test", os.path.basename(f)[len(k) + 1:-4], np.load(f))
             for f in sorted(glob.glob(os.path.join(P, f"{k}_*.npy")))] for k in ("oof", "st")}
write(os.path.join(ROOT, "src", "models", "imm_2ekf", "results", "imm_1hz"), items["oof"], items["st"])

# Model 2 (GNN), fixed window chosen on CV, with its temperature
for v in ("nomp", "mp"):
    R = os.path.join(ROOT, "src", "models", "road_gnn_rlcd", "results", f"m2_{v}")
    info = json.load(open(os.path.join(R, "run_info.json")))
    temps = json.load(open(os.path.join(R, "temperature.json")))
    w = [round(t["window_s"], 3) for t in temps.values()].index(info["chosen_window_s"])
    T = temps[str(w)]["T"]
    P = os.path.join(ROOT, "outputs", "preds", f"m2_{v}")
    its = {}
    for k, part in (("oof", "train"), ("st", "student_test")):
        its[k] = []
        for f in sorted(glob.glob(os.path.join(P, f"{k}_*.npz"))):
            tid = os.path.basename(f)[len(k) + 1:-4]
            ptr = prepare(load_trip(part, tid)).cands.ptr
            its[k].append((part, tid, flat_softmax(np.load(f)["logits"][w], ptr, T)))
    write(R, its["oof"], its["st"], {"window_s": info["chosen_window_s"]})
