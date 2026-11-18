"""Route-independent candidate generation (inference-safe).

For each GPS fix: all canonical directed arcs whose segment lies within RADIUS
metres, ranked by point-to-segment distance, keeping at most MAX_K. Both
directions of a two-way street are separate candidates (identical distance;
direction features separate them). If nothing is within RADIUS we retry with
FALLBACK_RADIUS and keep the FALLBACK_K nearest; if still nothing, the fix has
zero candidates (handled explicitly by the decoder and metrics).

Output: flat "candidate rows" (one per fix x arc), the format every scorer uses.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roadnet import RoadNet

RADIUS = 50.0
MAX_K = 16
FALLBACK_RADIUS = 200.0
FALLBACK_K = 4


@dataclass
class Candidates:
    obs: np.ndarray      # (R,) fix index
    arc: np.ndarray      # (R,) canonical arc id
    dist: np.ndarray     # (R,) point-to-segment distance (m)
    offset: np.ndarray   # (R,) distance along arc of the foot point (m)
    frac: np.ndarray     # (R,) offset / length
    foot: np.ndarray     # (R,2) foot point
    ptr: np.ndarray      # (T+1,) rows of fix t are ptr[t]:ptr[t+1]
    radius: float
    max_k: int

    def rows(self, t):
        return slice(self.ptr[t], self.ptr[t + 1])


def generate(rn: RoadNet, xy: np.ndarray, radius: float = RADIUS, max_k: int = MAX_K) -> Candidates:
    T = len(xy)
    pi, ai = rn.query_radius(xy, radius)
    have = np.zeros(T, bool)
    have[pi] = True
    miss = np.nonzero(~have)[0]
    if len(miss):
        pj, aj = rn.query_radius(xy[miss], FALLBACK_RADIUS)
        pi = np.concatenate([pi, miss[pj]])
        ai = np.concatenate([ai, aj])
        fb = np.zeros(len(pi), bool)
        fb[-len(pj):] = True if len(pj) else False
    else:
        fb = np.zeros(len(pi), bool)
    dist, off, frac, foot = rn.project_points(xy[pi], ai)
    order = np.lexsort((ai, dist, pi))
    pi, ai, dist, off, frac, foot, fb = pi[order], ai[order], dist[order], off[order], frac[order], foot[order], fb[order]
    # rank within fix
    start = np.searchsorted(pi, np.arange(T))
    rank = np.arange(len(pi)) - start[pi]
    keep = np.where(fb, rank < FALLBACK_K, rank < max_k)
    pi, ai, dist, off, frac, foot = pi[keep], ai[keep], dist[keep], off[keep], frac[keep], foot[keep]
    ptr = np.searchsorted(pi, np.arange(T + 1))
    return Candidates(pi, ai, dist, off, frac, foot, ptr, radius, max_k)


def recall_table(cands: Candidates, labels: np.ndarray):
    """Per-fix: is the derived label among candidates, and its distance rank."""
    T = len(cands.ptr) - 1
    hit = np.zeros(T, bool)
    rank = np.full(T, -1)
    is_lab = cands.arc == labels[cands.obs]
    idx = np.nonzero(is_lab)[0]
    hit[cands.obs[idx]] = True
    rank[cands.obs[idx]] = idx - cands.ptr[cands.obs[idx]]
    n = np.diff(cands.ptr)
    return hit, rank, n
