"""
PathSnap - run classifier-only vs classifier+HMM decoding and evaluate.

Usage (from the repo root):
    python -m src.hmm.run_decoder --scorer gaussian --part student_test

--scorer gaussian   Gaussian placeholder emission on dist_m (sigma from train).
--scorer <name>     Any other name reads classifier probabilities from
                    data/scores/<name>/<part>/<trip>.parquet (columns pt, arc_id, proba)
                    and uses log(clip(proba, 1e-9, 1)) as the emission score.

Outputs
    data/predictions/<scorer>/<part>/<trip>.parquet       per-point predictions (clf + hmm)
    data/predictions/<scorer>/<part>/<trip>_route.npy     reconstructed HMM route (arc ids)
    reports/hmm/<scorer>_<part>_metrics.csv               per-trip metrics, one row per method
    reports/hmm/<scorer>_<part>_summary.md                summary table
    reports/hmm/params_train.json                         sigma / beta estimated on train
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.data.preprocess import _read, load_trip
from src.evaluation import metrics as M
from src.hmm.scoring import estimate_params, gaussian_emission_logp, proba_to_logp, transition_logp
from src.hmm.viterbi import ArcGraph, argmax_decode, reconstruct_route, viterbi

SUMMARY_COLS = [
    ("top1_acc", "Top-1 acc (%)"), ("candidate_miss_rate", "Cand. miss (%)"),
    ("continuity_pct", "Continuity (%)"),
    ("dir_precision", "Dir P"), ("dir_recall", "Dir R"), ("dir_f1", "Dir F1"),
    ("dir_len_precision", "Dir len-P"), ("dir_len_recall", "Dir len-R"),
    ("undir_f1", "Undir F1"), ("undir_len_precision", "Undir len-P"), ("undir_len_recall", "Undir len-R"),
    ("route_gaps", "Route gaps"), ("n_chain_breaks", "Chain breaks"), ("decode_time_s", "Decode s"),
]


def emission_scores(em: pd.DataFrame, scorer: str, part: str, trip: str, params: dict, scores_dir: str):
    if scorer == "gaussian":
        return gaussian_emission_logp(em, params["sigma_m"])
    f = os.path.join(scores_dir, scorer, part, f"{trip}.parquet")
    if not os.path.exists(f):
        raise FileNotFoundError(f"no classifier scores for scorer '{scorer}': expected {f}")
    sc = pd.read_parquet(f)[["pt", "arc_id", "proba"]]
    p = em[["pt", "arc_id"]].merge(sc, on=["pt", "arc_id"], how="left").proba
    return proba_to_logp(p.fillna(0.0).values)


def evaluate(method, pred, route_full, gaps, trip, em, graph, extra):
    pts = trip.points
    true_arc, route_pos = pts.true_arc.to_numpy(), pts.route_pos.to_numpy()
    row = {"trip": trip.trip_id, "method": method, **M.pointwise(pred, true_arc, em),
           **M.continuity(pred, pts.segment.to_numpy(), trip.arcs), "route_gaps": gaps, **extra}
    win = M.labelled_window(true_arc, route_pos)
    if win is not None and len(trip.route):
        p0, p1, r0, r1 = win
        pred_route, _ = reconstruct_route(pred[p0:p1 + 1], graph)
        row.update(M.sequence(pred_route, trip.route[r0:r1 + 1], trip.arcs))
    row["full_route_arcs"] = len(route_full)
    return row


def run(scorer: str, part: str, processed_dir: str, features_dir: str, pred_dir: str,
        report_dir: str, scores_dir: str):
    params = estimate_params(features_dir, "train")
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "params_train.json"), "w") as fh:
        json.dump(params, fh, indent=1)
    print(f"params from train: sigma={params['sigma_m']:.2f} m, beta={params['beta_m']:.2f} m")

    out_dir = os.path.join(pred_dir, scorer, part)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for trip_id in sorted(os.listdir(os.path.join(processed_dir, part))):
        if not os.path.isdir(os.path.join(processed_dir, part, trip_id)):
            continue
        trip = load_trip(processed_dir, part, trip_id)
        em = _read(os.path.join(features_dir, part, f"{trip_id}_emission"))
        tr = _read(os.path.join(features_dir, part, f"{trip_id}_transition"))

        # decoder inputs: label-free columns only
        em_in = em[["pt", "arc_id"]].assign(emission_logp=emission_scores(em, scorer, part, trip_id, params, scores_dir))
        tr_in = tr[["pt_from", "pt_to", "arc_from", "arc_to"]].assign(transition_logp=transition_logp(tr, params["beta_m"]))
        pts_in = trip.points[["segment"]].reset_index(drop=True)
        graph = ArcGraph(trip.arcs, trip.nodes_xy)

        t0 = time.perf_counter()
        pred_clf = argmax_decode(em_in, len(pts_in))
        route_clf, gaps_clf = reconstruct_route(pred_clf, graph)
        t_clf = time.perf_counter() - t0

        t0 = time.perf_counter()
        res = viterbi(em_in, tr_in, pts_in)
        pred_hmm = res.points.pred_arc.to_numpy()
        route_hmm, gaps_hmm = reconstruct_route(pred_hmm, graph)
        t_hmm = time.perf_counter() - t0

        pd.DataFrame({"pt": np.arange(len(pts_in)), "segment": pts_in.segment.values, "t": trip.points.t.values,
                      "pred_arc_clf": pred_clf, "pred_arc_hmm": pred_hmm,
                      "hmm_chain_id": res.points.chain_id.values}).to_parquet(
            os.path.join(out_dir, f"{trip_id}.parquet"), index=False)
        np.save(os.path.join(out_dir, f"{trip_id}_route.npy"), route_hmm)

        rows.append(evaluate("classifier", pred_clf, route_clf, gaps_clf, trip, em, graph,
                             {"decode_time_s": t_clf, "n_chain_breaks": np.nan}))
        rows.append(evaluate("classifier+hmm", pred_hmm, route_hmm, gaps_hmm, trip, em, graph,
                             {"decode_time_s": t_hmm, "n_chain_breaks": res.n_chain_breaks,
                              "n_unreachable_breaks": res.n_unreachable_breaks, "n_chains": res.n_chains}))
        c, h = rows[-2], rows[-1]
        print(f"{part}/{trip_id}: acc {c['top1_acc']:.1f} -> {h['top1_acc']:.1f}  "
              f"continuity {c['continuity_pct']:.1f} -> {h['continuity_pct']:.1f}  "
              f"F1 {c.get('dir_f1', np.nan):.1f} -> {h.get('dir_f1', np.nan):.1f}  "
              f"breaks {res.n_chain_breaks}  {t_hmm:.2f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(report_dir, f"{scorer}_{part}_metrics.csv"), index=False)
    md = summary_md(df, scorer, part, params)
    with open(os.path.join(report_dir, f"{scorer}_{part}_summary.md"), "w") as fh:
        fh.write(md)
    print("\n" + md)
    return df


def summary_md(df: pd.DataFrame, scorer: str, part: str, params: dict) -> str:
    cols = [c for c, _ in SUMMARY_COLS if c in df]
    tab = df.groupby("method", sort=False)[cols].mean().rename(columns=dict(SUMMARY_COLS)).T.round(2)
    pooled = df.groupby("method", sort=False).apply(
        lambda g: (g.top1_acc * g.n_labelled).sum() / g.n_labelled.sum(), include_groups=False)
    w = df.pivot(index="trip", columns="method", values="continuity_pct")
    better = int((w["classifier+hmm"] > w["classifier"]).sum())
    hmm = df[df.method == "classifier+hmm"]
    clf_c, hmm_c = tab.loc["Continuity (%)", "classifier"], tab.loc["Continuity (%)", "classifier+hmm"]
    L = [f"# HMM decoding - scorer `{scorer}`, partition `{part}`\n",
         f"Trips: {df.trip.nunique()}. Emission: `{scorer}`"
         + (f" (sigma = {params['sigma_m']:.2f} m)" if scorer == "gaussian" else "")
         + f". Transition: Newson & Krumm exponential (beta = {params['beta_m']:.2f} m). "
           "Parameters estimated on train only.\n",
         "Values are means over trips (metrics in %, decode time in seconds per trip).\n",
         tab.to_markdown(), "",
         f"Pooled top-1 accuracy over all labelled points: classifier {pooled['classifier']:.2f}%, "
         f"classifier+HMM {pooled['classifier+hmm']:.2f}%.\n",
         f"**Continuity check:** classifier+HMM {hmm_c:.2f}% vs classifier-only {clf_c:.2f}% -> "
         f"{'HIGHER' if hmm_c > clf_c else 'NOT higher'}; HMM higher on {better}/{len(w)} trips.\n",
         f"Chain breaks: {int(hmm.n_chain_breaks.sum())} total "
         f"({int(hmm.n_unreachable_breaks.sum())} from unreachable steps). "
         f"Total HMM decode time {hmm.decode_time_s.sum():.1f} s.\n",
         "Notes: sequence metrics compare the reconstructed route (shortest-path filled) with the "
         "reference route between the first and last labelled point; 'Dir' = exact directed arc, "
         "'Undir' = arc or its reverse. Candidate miss rate is the ceiling on accuracy and is the same "
         "for both methods."]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scorer", default="gaussian")
    ap.add_argument("--part", default="student_test")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--pred-dir", default="data/predictions")
    ap.add_argument("--report-dir", default="reports/hmm")
    ap.add_argument("--scores-dir", default="data/scores")
    a = ap.parse_args()
    run(a.scorer, a.part, a.processed_dir, a.features_dir, a.pred_dir, a.report_dir, a.scores_dir)


if __name__ == "__main__":
    main()
