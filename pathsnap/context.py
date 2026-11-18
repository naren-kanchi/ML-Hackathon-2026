"""TripContext: everything derived from (.track, .nodes, .arcs) that scorers and
the decoder share. Inference-safe: never touches the route.

Scorer interface (for every teammate model):
    probs = scorer.predict_proba(ctx)   # np.ndarray, shape (len(ctx.cands.arc),)
probs[ctx.cands.rows(t)] must sum to 1 for every fix t that has candidates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .candidates import Candidates, generate
from .io import Trip
from .roadnet import RoadNet


@dataclass
class TripContext:
    trip: Trip
    rn: RoadNet
    xy: np.ndarray       # (T,2) projected metres
    t: np.ndarray        # (T,) seconds
    cands: Candidates

    @property
    def T(self):
        return len(self.t)


def prepare(trip: Trip) -> TripContext:
    rn = RoadNet(trip)
    xy = rn.project_track(trip.track)
    return TripContext(trip, rn, xy, trip.track[:, 2].astype(np.float64), generate(rn, xy))


def segment_softmax(logits: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    """Softmax within each fix's candidate block."""
    out = np.empty_like(logits, dtype=np.float64)
    for t in range(len(ptr) - 1):
        s, e = ptr[t], ptr[t + 1]
        if e > s:
            z = logits[s:e] - logits[s:e].max()
            ez = np.exp(z)
            out[s:e] = ez / ez.sum()
    return out


class DistanceScorer:
    """Untrained reference: p(a|z) ∝ exp(-d^2 / 2σ^2). Used to test the decoder."""
    name = "distance_only"

    def __init__(self, sigma=10.0):
        self.sigma = sigma

    def predict_proba(self, ctx: TripContext) -> np.ndarray:
        return segment_softmax(-ctx.cands.dist ** 2 / (2 * self.sigma ** 2), ctx.cands.ptr)
