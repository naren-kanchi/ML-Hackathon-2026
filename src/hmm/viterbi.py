"""
PathSnap - HMM / Viterbi route reconstruction.

Hidden state  = one candidate arc at one GPS point. Each point's candidates keep
                their own arc_id, so the decoded state maps straight back to an arc.
Emission      = column 'emission_logp' of the emission table (log-probability from
                any model: Gaussian placeholder, HistGradientBoosting, ...).
Transition    = column 'transition_logp' of the transition table (-inf = impossible).

Decoding is done in log space, chain by chain. A chain ends and a new one starts when:
  * the GPS segment changes (gaps split by preprocessing are decoded independently),
  * a point has no candidates (no prediction for it: arc -1),
  * every transition into the next point is impossible (an "unreachable break";
    the chain restarts at that point and the break is counted).

No labels are used here: the decoder only reads pt / arc_id / emission_logp,
pt_from / pt_to / arc_from / arc_to / transition_logp and the point segment.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import pandas as pd

NEG_INF = -np.inf


@dataclass
class DecodeResult:
    points: pd.DataFrame     # one row per GPS point: pt, segment, pred_arc, chain_id
    n_chains: int            # number of independently decoded chains
    n_unreachable_breaks: int  # restarts because every transition was impossible
    n_no_candidate_points: int  # points left without a prediction (arc -1)

    @property
    def n_chain_breaks(self) -> int:
        """Restarts inside a segment (unreachable steps + chains cut by points with no candidates)."""
        return self.n_chains - self.points.loc[self.points.pred_arc >= 0, "segment"].nunique()


def _candidate_tables(emission_df: pd.DataFrame):
    """Per point: candidate arc_ids and emission log-probs, in a fixed order."""
    em = emission_df[["pt", "arc_id", "emission_logp"]].sort_values(["pt", "emission_logp"],
                                                                    ascending=[True, False], kind="stable")
    em = em.reset_index(drop=True)
    em["cidx"] = em.groupby("pt").cumcount()
    logp = em.emission_logp.to_numpy(float)
    logp = np.where(np.isnan(logp), NEG_INF, logp)
    cand = {}
    starts = np.flatnonzero(np.r_[True, em.pt.values[1:] != em.pt.values[:-1]])
    ends = np.r_[starts[1:], len(em)]
    for s, e in zip(starts, ends):
        cand[int(em.pt.values[s])] = (em.arc_id.values[s:e].astype(np.int64), logp[s:e])
    return cand, em[["pt", "arc_id", "cidx"]]


def _transition_matrices(transition_df: pd.DataFrame, cidx: pd.DataFrame, cand: dict):
    """Dense (n_from x n_to) log-prob matrix for every consecutive point pair."""
    tr = transition_df[["pt_from", "pt_to", "arc_from", "arc_to", "transition_logp"]]
    tr = tr.merge(cidx.rename(columns={"pt": "pt_from", "arc_id": "arc_from", "cidx": "j"}),
                  on=["pt_from", "arc_from"], how="inner")
    tr = tr.merge(cidx.rename(columns={"pt": "pt_to", "arc_id": "arc_to", "cidx": "k"}),
                  on=["pt_to", "arc_to"], how="inner")
    lp = tr.transition_logp.to_numpy(float)
    lp = np.where(np.isnan(lp), NEG_INF, lp)
    mats = {}
    for (a, b), idx in tr.groupby(["pt_from", "pt_to"]).indices.items():
        M = np.full((len(cand[a][0]), len(cand[b][0])), NEG_INF)
        M[tr.j.values[idx], tr.k.values[idx]] = lp[idx]
        mats[(int(a), int(b))] = M
    return mats


def viterbi(emission_df: pd.DataFrame, transition_df: pd.DataFrame, points_df: pd.DataFrame) -> DecodeResult:
    """Decode one trip. points_df row i is GPS point pt=i and must have a 'segment' column."""
    n = len(points_df)
    seg = points_df["segment"].to_numpy()
    cand, cidx = _candidate_tables(emission_df)
    mats = _transition_matrices(transition_df, cidx, cand)

    pred = np.full(n, -1, np.int64)
    chain = np.full(n, -1, np.int64)
    n_chains = n_unreach = 0

    # current open chain
    chain_pts: list[int] = []        # points in the chain
    backs: list[np.ndarray] = []     # backpointers for chain_pts[1:]
    delta = None                     # best log score per candidate of chain_pts[-1]

    def close():
        nonlocal chain_pts, backs, delta, n_chains
        if not chain_pts:
            return
        k = int(np.argmax(delta))
        for idx in range(len(chain_pts) - 1, -1, -1):
            p = chain_pts[idx]
            pred[p] = cand[p][0][k]
            chain[p] = n_chains
            if idx:
                k = int(backs[idx - 1][k])
        n_chains += 1
        chain_pts, backs, delta = [], [], None

    def start(p):
        nonlocal chain_pts, delta
        e = cand[p][1]
        if not np.isfinite(e).any():          # no usable candidate at all
            return
        chain_pts, delta = [p], e.copy()

    for p in range(n):
        if p not in cand:                     # no candidates: break the chain, no prediction
            close()
            continue
        prev = chain_pts[-1] if chain_pts else None
        if prev is None or prev != p - 1 or seg[prev] != seg[p]:
            close()                           # new segment (or chain cut by a missing point)
            start(p)
            continue
        M = mats.get((prev, p))
        score = delta[:, None] + (M if M is not None else NEG_INF) + cand[p][1][None, :]
        best_from = np.argmax(score, axis=0)
        new_delta = score[best_from, np.arange(score.shape[1])]
        if not np.isfinite(new_delta).any():  # every transition impossible: restart here
            n_unreach += 1
            close()
            start(p)
            continue
        backs.append(best_from)
        chain_pts.append(p)
        delta = new_delta
    close()

    out = pd.DataFrame({"pt": np.arange(n), "segment": seg, "pred_arc": pred, "chain_id": chain})
    return DecodeResult(out, n_chains, n_unreach, int((pred < 0).sum()))


def argmax_decode(emission_df: pd.DataFrame, n_points: int) -> np.ndarray:
    """Classifier-only baseline: best-scoring candidate per point, -1 if none."""
    pred = np.full(n_points, -1, np.int64)
    em = emission_df[["pt", "arc_id", "emission_logp"]].copy()
    em["emission_logp"] = em.emission_logp.fillna(NEG_INF)
    best = em.sort_values(["pt", "emission_logp"], ascending=[True, False], kind="stable").drop_duplicates("pt")
    best = best[np.isfinite(best.emission_logp)]
    pred[best.pt.values] = best.arc_id.values
    return pred


# ----------------------------------------------------------------------------
# route reconstruction
# ----------------------------------------------------------------------------
class ArcGraph:
    """Directed road graph for shortest-path filling (Dijkstra with parent pointers)."""

    def __init__(self, arcs: pd.DataFrame, nodes_xy: np.ndarray):
        a = arcs.reset_index() if "arc_id" not in arcs.columns else arcs
        order = np.argsort(a.u.values, kind="stable")
        self.v = a.v.values[order]
        self.w = a.length_m.values[order]
        self.aid = a.arc_id.values[order]
        self.ptr = np.r_[0, np.cumsum(np.bincount(a.u.values, minlength=len(nodes_xy)))]
        self.uv = dict(zip(a.arc_id.values, zip(a.u.values, a.v.values)))
        self.nodes_xy = nodes_xy

    def shortest_arcs(self, s: int, t: int, cutoff: float):
        """Arc ids of the shortest directed path node s -> node t, or None if none within cutoff."""
        if s == t:
            return []
        dist, parent = {s: 0.0}, {}
        heap = [(0.0, s)]
        while heap:
            d, x = heapq.heappop(heap)
            if x == t:
                path = []
                while x != s:
                    arc, x = parent[x]
                    path.append(arc)
                return path[::-1]
            if d > dist.get(x, np.inf):
                continue
            for k in range(self.ptr[x], self.ptr[x + 1]):
                y, nd = int(self.v[k]), d + self.w[k]
                if nd <= cutoff and nd < dist.get(y, np.inf):
                    dist[y] = nd
                    parent[y] = (int(self.aid[k]), x)
                    heapq.heappush(heap, (nd, y))
        return None


def reconstruct_route(pred_arcs, graph: ArcGraph, cutoff_min_m: float = 1000.0, cutoff_factor: float = 3.0):
    """Turn per-point arcs into a connected arc sequence.

    Repeated arcs collapse to one entry; between two different consecutive arcs A -> B
    the shortest directed path from A's end node to B's start node is inserted.
    Points with arc -1 are skipped. Dijkstra is bounded by
    max(cutoff_min_m, cutoff_factor * straight-line distance); if no path is found
    B is appended anyway and the jump is counted as a gap.
    Returns (route arc ids, number of gaps)."""
    seq = [int(a) for a in pred_arcs if a >= 0]
    route, gaps = [], 0
    for a in seq:
        if not route:
            route.append(a)
            continue
        last = route[-1]
        if a == last:
            continue
        s, t = graph.uv[last][1], graph.uv[a][0]
        if s != t:
            gc = float(np.hypot(*(graph.nodes_xy[t] - graph.nodes_xy[s])))
            path = graph.shortest_arcs(s, t, max(cutoff_min_m, cutoff_factor * gc))
            if path is None:
                gaps += 1
            else:
                route.extend(path)
        route.append(a)
    return np.array(route, np.int64), gaps
