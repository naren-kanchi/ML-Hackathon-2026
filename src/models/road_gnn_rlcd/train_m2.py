"""Train + evaluate Model 2 (road-graph scorer), variants:
   nomp : geometry + GPS history, arc encoder WITHOUT message passing (A)
   mp   : same scorer WITH 2 layers of directed message passing (B)
Same frozen folds, seeds, epochs, optimiser for both. All three history windows
are trained jointly (loss = mean CE over windows).

Outputs per variant (results/m2_<v>/ in this folder, outputs/preds/m2_<v>/):
  cv_history.csv           per-epoch train / val loss per fold and window
  oof_<trip>.npz           OOF logits for every window (used by RLCD)
  st_<trip>.npz            student-test logits (final refit) for every window
  temperature.json         one temperature per window fit on OOF
  window_table.csv         classifier-only metrics per window (CV OOF + student-test)
  summary.csv              classifier-only and +HMM for the chosen fixed window
artifacts/model2_<v>.pt
python -m src.models.road_gnn_rlcd.train_m2 [nomp|mp ...]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pathsnap.evaluate import evaluate_many, summarise  # noqa: E402
from pathsnap.io import list_trips  # noqa: E402
from pathsnap.listwise import fit_temperature  # noqa: E402
from src.models.road_gnn_rlcd.model import WINDOWS_S, RoadScorer, segment_log_softmax, to_torch  # noqa: E402
from pathsnap.splits import SEED, load_folds  # noqa: E402

CACHE = os.path.join(ROOT, "outputs", "cache", "m2")
EPOCHS, LR, WD, MAX_FIX, CHUNK = 15, 2e-3, 1e-4, 4000, 1000
nW = len(WINDOWS_S)
torch.set_num_threads(8)
_cache = {}


def load(part, tid):
    key = (part, tid)
    if key not in _cache:
        d = torch.load(os.path.join(CACHE, part, f"{tid}.pt"), weights_only=False)
        _cache[key] = (to_torch(d), d)
    return _cache[key]


def trip_loss(model, td, raw, fixes, train_mask=True):
    y = raw["y_row"][fixes]
    ok = y >= 0
    if train_mask:
        ok &= ~raw["far"][fixes]
    fixes, y = fixes[ok], y[ok]
    if len(fixes) == 0:
        return None, 0
    ptr = raw["ptr"]
    rows = np.concatenate([np.arange(ptr[f], ptr[f + 1]) for f in fixes])
    rows_t = torch.from_numpy(rows)
    emb = model.arc_embed(td)
    td2 = dict(td, emb=emb)
    obs = td["obs"][rows_t]
    tot = 0.0
    for w in range(nW):
        z = model(td2, rows_t, w)
        ls = segment_log_softmax(z, obs, raw["T"])
        tot = tot - ls[torch.from_numpy(np.searchsorted(rows, y))].mean()
    return tot / nW, len(fixes)


@torch.no_grad()
def trip_logits(model, td, raw):
    model.eval()
    rows_t = torch.arange(len(raw["obs"]))
    td2 = dict(td, emb=model.arc_embed(td))
    return np.stack([model(td2, rows_t, w).numpy() for w in range(nW)])   # (nW, R)


@torch.no_grad()
def val_losses(model, trips):
    tot, n = np.zeros(nW), 0
    for t in trips:
        td, raw = load("train", t)
        Z = trip_logits(model, td, raw)
        y = raw["y_row"]
        ok = (y >= 0) & ~raw["far"]
        for w in range(nW):
            ls = segment_log_softmax(torch.from_numpy(Z[w]), td["obs"], raw["T"]).numpy()
            tot[w] += -ls[y[ok]].sum()
        n += ok.sum()
    return tot / n


def train(variant, trips, val_trips=None, seed=SEED, log=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = RoadScorer(message_passing=(variant == "mp"))
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        tl, tn = 0.0, 0
        for t in rng.permutation(trips):
            td, raw = load("train", t)
            T = raw["T"]
            sel = rng.choice(T, min(T, MAX_FIX), replace=False)
            for c0 in range(0, len(sel), CHUNK):
                loss, n = trip_loss(model, td, raw, np.sort(sel[c0:c0 + CHUNK]))
                if loss is None:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
                tl += loss.item() * n
                tn += n
        rec = dict(epoch=ep + 1, train_loss=tl / tn)
        if val_trips:
            vl = val_losses(model, val_trips)
            rec.update({f"val_loss_w{w}": vl[w] for w in range(nW)})
        if log is not None:
            log.append(rec)
        print(variant, rec, flush=True)
    return model, time.time() - t0


def to_padded(Z, ptr, K=16):
    T = len(ptr) - 1
    n = np.diff(ptr)
    fix = np.repeat(np.arange(T), n)
    slot = np.arange(len(fix)) - ptr[fix]
    P = np.zeros((T, K), np.float32)
    M = np.zeros((T, K), bool)
    P[fix, slot] = Z
    M[fix, slot] = True
    return P, M


def flat_softmax(z, ptr, T=1.0):
    out = np.zeros_like(z, dtype=np.float64)
    for f in range(len(ptr) - 1):
        s, e = ptr[f], ptr[f + 1]
        if e > s:
            a = z[s:e] / T
            a = np.exp(a - a.max())
            out[s:e] = a / a.sum()
    return out


def quick_metrics(probs_by_trip, part):
    """Classifier-only per-window metrics (no decoder): top1, NLL, Brier over all fixes."""
    rows = []
    for t, P in probs_by_trip.items():
        _, raw = load(part, t)
        ptr, y = raw["ptr"], raw["y_row"]
        T = raw["T"]
        for w in range(nW):
            p = P[w]
            top = np.full(T, -1)
            for f in range(T):
                if ptr[f + 1] > ptr[f]:
                    top[f] = ptr[f] + int(np.argmax(p[ptr[f]:ptr[f + 1]]))
            pl = np.where(y >= 0, p[np.maximum(y, 0)], 0.0)
            yv = np.zeros(len(p))
            yv[y[y >= 0]] = 1
            br = np.bincount(raw["obs"], weights=(p - yv) ** 2, minlength=T) + (y < 0)
            rows.append(dict(trip=t, window=WINDOWS_S[w], n_fix=T, top1=float(np.mean((top == y) & (y >= 0))),
                             log_loss=float(np.mean(-np.log(np.clip(pl, 1e-6, 1)))), brier=float(br.mean())))
    df = pd.DataFrame(rows)
    return df.groupby("window").apply(lambda g: pd.Series(
        {c: np.average(g[c], weights=g.n_fix) for c in ["top1", "log_loss", "brier"]}), include_groups=False).reset_index()


def run(variant):
    RES = os.path.join(HERE, "results", f"m2_{variant}")
    PRED = os.path.join(ROOT, "outputs", "preds", f"m2_{variant}")
    os.makedirs(RES, exist_ok=True)
    os.makedirs(PRED, exist_ok=True)
    folds = load_folds()["folds"]
    hist, oof = [], {}
    t_cv = time.time()
    for k in range(len(folds)):
        tr = [t for j, f in enumerate(folds) if j != k for t in f]
        log = []
        model, _ = train(variant, tr, folds[k], seed=SEED + k, log=log)
        hist += [dict(fold=k, **r) for r in log]
        for t in folds[k]:
            td, raw = load("train", t)
            oof[t] = trip_logits(model, td, raw)
    cv_time = time.time() - t_cv
    pd.DataFrame(hist).to_csv(os.path.join(RES, "cv_history.csv"), index=False)
    # per-window temperature on OOF
    temps = {}
    for w in range(nW):
        Zs, Ms, ys = [], [], []
        for t, Z in oof.items():
            _, raw = load("train", t)
            P, M = to_padded(Z[w], raw["ptr"])
            y = np.where((raw["y_row"] >= 0) & ~raw["far"], raw["y_row"] - raw["ptr"][:-1], -1)
            Zs.append(P); Ms.append(M); ys.append(y)
        T, nT, n1 = fit_temperature(np.concatenate(Zs), np.concatenate(Ms), np.concatenate(ys))
        temps[w] = dict(window_s=WINDOWS_S[w], T=T, oof_nll_T1=n1, oof_nll_T=nT)
    json.dump(temps, open(os.path.join(RES, "temperature.json"), "w"), indent=1)
    # final refit on all training trips
    all_tr = sorted(sum(folds, []))
    final, ft = train(variant, all_tr, seed=SEED)
    torch.save(dict(state=final.state_dict(), variant=variant, windows_s=WINDOWS_S,
                    temperatures={w: temps[w]["T"] for w in temps}, epochs=EPOCHS, lr=LR, wd=WD, seed=SEED),
               os.path.join(HERE, "artifacts", f"model2_{variant}.pt"))
    st = {}
    t_inf = time.time()
    for t in list_trips("student_test"):
        td, raw = load("student_test", t)
        st[t] = trip_logits(final, td, raw)
    inf_time = time.time() - t_inf
    for t, Z in oof.items():
        np.savez_compressed(os.path.join(PRED, f"oof_{t}.npz"), logits=Z)
    for t, Z in st.items():
        np.savez_compressed(os.path.join(PRED, f"st_{t}.npz"), logits=Z)
    # classifier-only per-window metrics (raw = T 1, cal = temperature)
    tabs = []
    for name, src, part in [("cv_oof", oof, "train"), ("student_test", st, "student_test")]:
        for cal in (False, True):
            probs = {t: np.stack([flat_softmax(Z[w], load(part, t)[1]["ptr"], temps[w]["T"] if cal else 1.0)
                                  for w in range(nW)]) for t, Z in src.items()}
            q = quick_metrics(probs, part)
            q.insert(0, "calibrated", cal)
            q.insert(0, "split", name)
            tabs.append(q)
    wt = pd.concat(tabs)
    wt.to_csv(os.path.join(RES, "window_table.csv"), index=False)
    print(wt.round(4).to_string(index=False))
    # fixed window chosen on CV OOF calibrated log loss (training data only)
    cvcal = wt[(wt.split == "cv_oof") & wt.calibrated]
    wbest = int(np.argmin(cvcal.log_loss.values))
    Tb = temps[wbest]["T"]
    items_cv = [("train", t, flat_softmax(Z[wbest], load("train", t)[1]["ptr"], Tb)) for t, Z in oof.items()]
    items_st = [("student_test", t, flat_softmax(Z[wbest], load("student_test", t)[1]["ptr"], Tb)) for t, Z in st.items()]
    cv = pd.DataFrame(evaluate_many(items_cv))
    stt = pd.DataFrame(evaluate_many(items_st))
    cv.to_csv(os.path.join(RES, "cv_per_trip.csv"), index=False)
    stt.to_csv(os.path.join(RES, "student_test_per_trip.csv"), index=False)
    a, b = summarise(cv), summarise(stt)
    a.insert(0, "split", "cv_oof")
    b.insert(0, "split", "student_test")
    summ = pd.concat([a, b])
    summ.insert(1, "window_s", WINDOWS_S[wbest])
    summ.to_csv(os.path.join(RES, "summary.csv"), index=False)
    json.dump(dict(variant=variant, params=sum(p.numel() for p in final.parameters()), cv_wall_s=cv_time,
                   final_train_s=ft, st_scoring_s=inf_time, chosen_window_s=WINDOWS_S[wbest]),
              open(os.path.join(RES, "run_info.json"), "w"), indent=1)
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    for v in (sys.argv[1:] or ["nomp", "mp"]):
        run(v)
