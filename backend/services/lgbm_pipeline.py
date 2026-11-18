"""Selected model for the API: team preprocessing -> tuned LightGBM LambdaRank ->
tuned HMM/Viterbi (pathsnap/hmm.py).

Leakage guard: inference runs on a temporary copy of the trip that contains ONLY
.track, .nodes and .arcs, so preprocessing cannot see a .route file. The
reference route is read afterwards, in evaluate(), for aggregate metrics only;
no label or route id is ever returned.

Response schema matches backend/services/map_matching.py (the previous
HistGradientBoosting service) so the frontend keeps working. Note that the
observations are the team-preprocessed points (stationary runs collapsed,
points thinned to >= 10 m), not every raw 1 Hz fix.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

from pathsnap.hmm import HMMParams, classifier_only, reconstruct_route, viterbi
from pathsnap.io import Trip, load_trip_from_files
from pathsnap.metrics import continuity, macro_f1_trip, route_metrics
from pathsnap.roadnet import RoadNet
from src.data.preprocess import PreprocessConfig, preprocess_trip
from src.features.build_features import FeatureConfig, generate_candidates
from src.models.imm_2ekf.teamdata import ctx_from_candidate_lists

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "artifacts" / "models" / "lgbm_ranker_tuned"


@lru_cache(maxsize=1)
def load_bundle(path: str | None = None) -> dict[str, Any]:
    d = Path(path) if path else ARTIFACT
    cfg = json.loads((d / "config.json").read_text())
    return dict(cfg=cfg, booster=lgb.Booster(model_file=str(d / cfg["booster"])),
                hmm=HMMParams(**cfg["hmm_params"]), T=float(cfg["temperature"]))


def _group_softmax(score, pt, T):
    p = np.zeros(len(score))
    for g in np.unique(pt):
        m = pt == g
        z = score[m] / T
        e = np.exp(z - z.max())
        p[m] = e / e.sum()
    return p


def _inference_copy(trip_dir: Path, tmp: Path) -> Path:
    tid = trip_dir.name
    dst = tmp / "trip" / tid
    dst.mkdir(parents=True)
    for ext in ("track", "nodes", "arcs"):          # deliberately NOT .route
        shutil.copy2(trip_dir / f"{tid}.{ext}", dst / f"{tid}.{ext}")
    return dst


def predict_trip(trip_dir: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    tid = trip_dir.name
    with tempfile.TemporaryDirectory() as tmp:
        src = _inference_copy(trip_dir, Path(tmp))
        tp = preprocess_trip(str(src), tid, "inference", PreprocessConfig(**bundle["cfg"]["preprocess_config"]))
        raw = load_trip_from_files(*(str(src / f"{tid}.{e}") for e in ("track", "nodes", "arcs")), trip_id=tid)
    cands = generate_candidates(tp, FeatureConfig(**bundle["cfg"]["feature_config"]))
    cands = cands.sort_values(["pt", "dist_m"], kind="stable").reset_index(drop=True)
    X = cands[bundle["cfg"]["features"]].astype(np.float32).values
    score = bundle["booster"].predict(X)
    proba = _group_softmax(score, cands.pt.values, bundle["T"])

    pts = tp.points
    n = len(pts)
    ptr = np.searchsorted(cands.pt.values, np.arange(n + 1))
    rn = RoadNet(raw)
    trip = Trip(tid, np.column_stack([pts.lon.values, pts.lat.values, pts.t.values]), raw.nodes, raw.arcs)
    ctx, pr = ctx_from_candidate_lists(trip, rn, [cands.arc_id.values[ptr[i]:ptr[i + 1]] for i in range(n)],
                                       [proba[ptr[i]:ptr[i + 1]] for i in range(n)])
    hp = bundle["hmm"]
    clf, dec = classifier_only(ctx, pr), viterbi(ctx, pr, hp)
    route, breaks = reconstruct_route(ctx, dec, hp)
    clf_route, _ = reconstruct_route(ctx, clf, hp)

    def conf(d):
        return [float(pr[r]) if r >= 0 else 0.0 for r in d["row"]]

    def geometry(ids):
        return [{"arc_id": int(a), "coordinates": raw.nodes[raw.arcs[a, :2], :2].tolist()} for a in ids]

    dconf = conf(dec)
    observations = [{"timestamp": float(pts.t.values[i]), "gps": [float(pts.lon.values[i]), float(pts.lat.values[i])],
                     "arc_id": int(dec["arc"][i]), "confidence": dconf[i], "classifier_arc_id": int(clf["arc"][i])}
                    for i in range(n)]
    return {
        "trip_id": tid,
        "model": bundle["cfg"]["model"],
        "observation_count": int(n),
        "raw_observation_count": int(len(raw.track)),
        "predicted_arc_ids": [int(a) for a in dec["arc"]],
        "classifier_arc_ids": [int(a) for a in clf["arc"]],
        "observations": observations,
        "predicted_route": geometry(route),
        "classifier_route": geometry(clf_route),
        "mean_emission_confidence": float(np.mean([c for c in dconf if c > 0])) if any(dconf) else 0.0,
        "route_continuity": float(continuity(ctx, dec["arc"], hp, dec["netdist"])),
        "forced_restarts": int(sum(1 for _, r in dec["restarts"] if r != "start")),
        "route_breaks": int(breaks),
        "candidate_miss_rate": None,
        "evaluation": None,
        "_metadata": dict(ctx=ctx, rn=rn, dec=dec, clf=clf, route=route),
    }


def evaluate(prediction: dict[str, Any], trip_dir: Path) -> dict[str, Any] | None:
    """Aggregate metrics against the server-side reference route (after inference).
    Labels: the team's monotone route alignment on the same preprocessed points."""
    tid = trip_dir.name
    if not (trip_dir / f"{tid}.route").is_file():
        return None
    meta = prediction["_metadata"]
    rn = meta["rn"]
    tp = preprocess_trip(str(trip_dir), tid, "evaluation", PreprocessConfig())
    lab = tp.points.true_arc.values.astype(int)
    lab = np.where(lab >= 0, rn.canon[np.maximum(lab, 0)], -1)
    L = lab >= 0
    if not L.any():
        return {"evaluated_observations": 0}
    a_h, a_c = np.asarray(meta["dec"]["arc"]), np.asarray(meta["clf"]["arc"])
    c = meta["ctx"].cands
    covered = np.zeros(len(lab), bool)
    covered[c.obs[c.arc == lab[c.obs]]] = True
    rp = tp.points.route_pos.values[L]
    rm = route_metrics(rn, np.asarray(meta["route"]), rn.canon[tp.route], (int(rp.min()), int(rp.max())))

    def prf(pred):
        y, p = lab[L], pred[L]
        cls = np.union1d(np.unique(y), np.unique(p[p >= 0]))
        P = [np.sum((p == k) & (y == k)) / max(np.sum(p == k), 1) for k in cls]
        R = [np.sum((p == k) & (y == k)) / max(np.sum(y == k), 1) for k in cls]
        return float(np.mean(P)), float(np.mean(R))
    vp, vr = prf(a_h)
    return {
        "evaluated_observations": int(L.sum()),
        "candidate_miss_rate": float(1 - covered[L].mean()),
        "classifier_top1_accuracy": float((a_c[L] == lab[L]).mean()),
        "classifier_macro_f1": macro_f1_trip(a_c[L], lab[L]),
        "viterbi_top1_accuracy": float((a_h[L] == lab[L]).mean()),
        "viterbi_macro_f1": macro_f1_trip(a_h[L], lab[L]),
        "viterbi_precision_macro": vp,
        "viterbi_recall_macro": vr,
        "viterbi_route_continuity": prediction["route_continuity"],
        "route_f1": float(rm["route_f1_trim"]),
    }
