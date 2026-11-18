"""Model 1 (variant C): raw GPS features + IMM (straight + coordinated-turn) tracker
features -> compact listwise MLP candidate scorer.

Inference uses only (.track, .nodes, .arcs):
    ctx = prepare(trip); X = model1_features(ctx); probs = scorer.predict_proba(ctx)
"""
from __future__ import annotations

import json
import os

import numpy as np

from pathsnap.features_raw import RAW_NAMES, raw_features
from .trackers import TRACK_NAMES, TrackerParams, run_tracker, tracker_features

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FEATURE_NAMES = RAW_NAMES + ["imm_" + n for n in TRACK_NAMES] + ["imm_mu_turn"]


def tracker_params() -> TrackerParams:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "tracker_1hz.json")
    if os.path.exists(p):
        return TrackerParams(**json.load(open(p))["params"])
    return TrackerParams()


def model1_features(ctx, tp: TrackerParams | None = None) -> np.ndarray:
    tp = tp or tracker_params()
    tr = run_tracker(ctx.xy, ctx.t, "imm", tp)
    return np.hstack([raw_features(ctx), tracker_features(ctx, tr, imm=True)])
