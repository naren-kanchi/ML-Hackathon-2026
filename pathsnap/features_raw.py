"""Variant A features: raw GPS-derived candidate features (causal: fixes <= t only).

Per candidate row (fix t, arc a):
  dist            point-to-segment distance (m)
  dist_excess     dist - min dist among the fix's candidates
  rank            distance rank within the fix (0 = nearest)
  n_cands         number of candidates of the fix
  frac            foot position along the arc (0..1)
  log_len         log(1 + arc length)
  has_reverse     arc has an opposite-direction twin (two-way street)
  out_deg, in_deg branching at the arc's end / start node
  raw_cos         cos(raw GPS heading - arc heading)     (0 if heading unreliable)
  raw_sin_abs     |sin(...)|                              (0 if unreliable)
  raw_rel         1 if the raw heading is reliable (displacement >= MIN_DISP_M within MAX_LOOKBACK_S)
  raw_speed       displacement / elapsed time (m/s) over the same look-back
  raw_turn        heading change between the last two displacement vectors (rad, signed)
  dt_prev         seconds since previous fix (capped)
Raw heading is derived information (difference of GPS positions), not a sensor.
"""
from __future__ import annotations

import numpy as np

MIN_DISP_M = 5.0
MAX_LOOKBACK_S = 10.0

RAW_NAMES = ["dist", "dist_excess", "rank", "n_cands", "frac", "log_len", "has_reverse", "out_deg", "in_deg",
             "raw_cos", "raw_sin_abs", "raw_rel", "raw_speed", "raw_turn", "dt_prev"]


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def raw_heading(xy: np.ndarray, t: np.ndarray):
    """Causal heading: from the most recent earlier fix >= MIN_DISP_M away, within MAX_LOOKBACK_S."""
    T = len(t)
    head = np.zeros(T)
    rel = np.zeros(T, bool)
    speed = np.zeros(T)
    j = 0
    for i in range(1, T):
        # earliest index allowed
        while t[i] - t[j] > MAX_LOOKBACK_S:
            j += 1
        k = i - 1
        found = -1
        while k >= j:
            if np.hypot(*(xy[i] - xy[k])) >= MIN_DISP_M:
                found = k
                break
            k -= 1
        if found >= 0:
            d = xy[i] - xy[found]
            head[i] = np.arctan2(d[1], d[0])
            rel[i] = True
            speed[i] = np.hypot(*d) / (t[i] - t[found])
        else:
            kk = max(i - 1, 0)
            speed[i] = np.hypot(*(xy[i] - xy[kk])) / max(t[i] - t[kk], 1e-6)
    turn = np.zeros(T)
    ok = rel.copy()
    ok[1:] &= rel[:-1]
    turn[1:] = np.where(ok[1:], wrap(head[1:] - head[:-1]), 0.0)
    return head, rel, speed, turn


def raw_features(ctx) -> np.ndarray:
    c, rn = ctx.cands, ctx.rn
    o = c.obs
    T = ctx.T
    n = np.diff(c.ptr)
    mind = np.array([c.dist[c.ptr[t]] if n[t] else np.nan for t in range(T)])  # rows sorted by dist
    rank = np.arange(len(o)) - c.ptr[o]
    head, rel, speed, turn = raw_heading(ctx.xy, ctx.t)
    ang = wrap(head[o] - rn.heading[c.arc])
    r = rel[o]
    dt_prev = np.diff(ctx.t, prepend=ctx.t[0])
    F = np.column_stack([
        c.dist, c.dist - mind[o], rank, n[o], c.frac, np.log1p(rn.length[c.arc]), (rn.reverse[c.arc] >= 0),
        rn.out_degree_node(rn.to[c.arc]), rn.in_degree_node(rn.fr[c.arc]),
        np.where(r, np.cos(ang), 0.0), np.where(r, np.abs(np.sin(ang)), 0.0), r, np.minimum(speed[o], 60),
        turn[o], np.minimum(dt_prev[o], 60),
    ]).astype(np.float32)
    return F
