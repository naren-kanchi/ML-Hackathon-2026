"""Model 2: local road-graph neural network candidate scorer with GPS-history windows.

What the model sees (inference-safe: .track, .nodes, .arcs only)
-----------------------------------------------------------------
Road graph (per trip, cached once per trip):
  nodes of the GNN = canonical directed arcs within 2 hops of any candidate arc
  arc input features (no ids, no absolute heading -> rotation invariant):
      log(1+length), has_reverse, out-degree at end node, in-degree at start node
  directed edges a -> b when to(a) == from(b), with edge features
      cos/sin of the turn angle wrap(heading_b - heading_a), is_u_turn, log(1+len_b)
  Message passing (variant B): per layer
      h_a' = ReLU( W h_a + mean_{b in succ(a)} phi_s([h_b, e_ab]) + mean_{p in pred(a)} phi_p([h_p, e_pa]) )
  Variant A ("nomp") uses the same arc encoder with the message terms removed.
Per candidate (fix t, arc a, window w):
  geometry: dist, dist_excess, rank, n_cands, frac, log_len
  GPS history: up to H fixes from (t - W_w, t], never across a gap > GAP_S,
      expressed in the CANDIDATE ARC'S FRAME (rotate by -heading_a, origin at the
      foot point, /50 m) + mask -> small MLP. A track bending into a ramp looks
      curved in the motorway's frame and straight in the ramp's frame.
Score: MLP([arc_embedding, geometry, history_code, window one-hot]) -> logit;
masked softmax over the fix's candidates (variable K) -> ONE distribution per window.
Training: cross-entropy of the derived label for all three windows jointly, so
every window/action is usable later by the RLCD policy.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from pathsnap.context import TripContext

WINDOWS_S = (0.0, 5.0, 15.0)     # current fix / short / long (see report for derivation)
H = 8                            # max history points per window
GAP_S = 10.0                     # windows never cross a time gap > GAP_S
SCALE = 50.0


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def history(xy, t):
    """hist[T, nW, H, 2] offsets of past fixes relative to fix t (world frame, m); mask."""
    T = len(t)
    nW = len(WINDOWS_S)
    hist = np.zeros((T, nW, H, 2), np.float32)
    mask = np.zeros((T, nW, H), bool)
    seg_start = np.zeros(T, np.int64)       # first index after the most recent gap
    for i in range(1, T):
        seg_start[i] = i if t[i] - t[i - 1] > GAP_S else seg_start[i - 1]
    for w, W in enumerate(WINDOWS_S):
        if W <= 0:
            continue
        lo = np.searchsorted(t, t - W, side="left")
        lo = np.maximum(lo, seg_start)
        for i in range(T):
            idx = np.arange(lo[i], i)        # strictly past fixes within window
            if len(idx) == 0:
                continue
            if len(idx) > H:
                idx = idx[np.round(np.linspace(0, len(idx) - 1, H)).astype(int)]
            hist[i, w, :len(idx)] = xy[idx] - xy[i]
            mask[i, w, :len(idx)] = True
    return hist, mask


def build_trip_tensors(ctx: TripContext, hops: int = 2) -> dict:
    rn, c = ctx.rn, ctx.cands
    cand_arcs = np.unique(c.arc)
    S = set(cand_arcs.tolist())
    frontier = set(S)
    for _ in range(hops):
        nxt = set()
        for a in frontier:
            nxt.update(rn.successors(a).tolist())
            nxt.update(rn.predecessors(a).tolist())
        nxt -= S
        S |= nxt
        frontier = nxt
    S = np.array(sorted(S))
    loc = {a: i for i, a in enumerate(S)}
    src, dst = [], []
    for a in S:
        for b in rn.successors(a):
            if b in loc:
                src.append(loc[a]); dst.append(loc[b])
    src, dst = np.array(src, np.int64), np.array(dst, np.int64)
    A, B = S[src], S[dst]
    turn = _wrap(rn.heading[B] - rn.heading[A])
    efeat = np.column_stack([np.cos(turn), np.sin(turn), (rn.reverse[A] == B), np.log1p(rn.length[B])]).astype(np.float32)
    afeat = np.column_stack([np.log1p(rn.length[S]), rn.reverse[S] >= 0,
                             rn.out_degree_node(rn.to[S]), rn.in_degree_node(rn.fr[S])]).astype(np.float32)
    n = np.diff(c.ptr)
    rank = np.arange(len(c.obs)) - c.ptr[c.obs]
    mind = c.dist[c.ptr[:-1][c.obs]]
    geo = np.column_stack([c.dist / SCALE, (c.dist - mind) / SCALE, rank, n[c.obs], c.frac,
                           np.log1p(rn.length[c.arc])]).astype(np.float32)
    th = rn.heading[c.arc]
    rot = np.column_stack([np.cos(th), np.sin(th)]).astype(np.float32)
    foot_off = ((c.foot - ctx.xy[c.obs]) / SCALE).astype(np.float32)
    hist, hmask = history(ctx.xy, ctx.t)
    return dict(S=S, afeat=afeat, src=src, dst=dst, efeat=efeat,
                cand_loc=np.array([loc[a] for a in c.arc], np.int64), obs=c.obs, ptr=c.ptr,
                geo=geo, rot=rot, foot_off=foot_off, hist=(hist / SCALE).astype(np.float32), hmask=hmask)


# ---------------------------------------------------------------- model
class GNNLayer(nn.Module):
    def __init__(self, h, e):
        super().__init__()
        self.self_ = nn.Linear(h, h)
        self.succ = nn.Sequential(nn.Linear(h + e, h), nn.ReLU())
        self.pred = nn.Sequential(nn.Linear(h + e, h), nn.ReLU())

    def forward(self, x, src, dst, ef):
        n = x.shape[0]
        out = self.self_(x)
        deg_s = torch.zeros(n).index_add_(0, src, torch.ones(len(src))).clamp(min=1)[:, None]
        deg_p = torch.zeros(n).index_add_(0, dst, torch.ones(len(dst))).clamp(min=1)[:, None]
        m_s = self.succ(torch.cat([x[dst], ef], 1))           # message from successor b to a=src
        out = out + torch.zeros_like(out).index_add_(0, src, m_s) / deg_s
        m_p = self.pred(torch.cat([x[src], ef], 1))           # message from predecessor a to b=dst
        out = out + torch.zeros_like(out).index_add_(0, dst, m_p) / deg_p
        return torch.relu(out)


class RoadScorer(nn.Module):
    def __init__(self, message_passing=True, h=32, layers=2, hist_h=32, mlp_h=64):
        super().__init__()
        self.mp = message_passing
        self.enc = nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.layers = nn.ModuleList([GNNLayer(h, 4) for _ in range(layers)]) if message_passing else None
        # same parameter budget order for nomp: extra per-arc MLP layers instead of messages
        self.post = None if message_passing else nn.Sequential(*[m for _ in range(layers) for m in (nn.Linear(h, h), nn.ReLU())])
        self.hist = nn.Sequential(nn.Linear(H * 3 + 2, hist_h), nn.ReLU(), nn.Linear(hist_h, hist_h), nn.ReLU())
        nW = len(WINDOWS_S)
        self.head = nn.Sequential(nn.Linear(h + 6 + hist_h + nW, mlp_h), nn.ReLU(), nn.Dropout(0.1),
                                  nn.Linear(mlp_h, mlp_h), nn.ReLU(), nn.Linear(mlp_h, 1))

    def arc_embed(self, d):
        x = self.enc(d["afeat"])
        if self.mp:
            for L in self.layers:
                x = L(x, d["src"], d["dst"], d["efeat"])
        else:
            x = self.post(x)
        return x

    def forward(self, d, rows, w):
        """Logits for candidate rows `rows` (LongTensor) under window index w."""
        emb = self.arc_embed(d) if "emb" not in d else d["emb"]
        e = emb[d["cand_loc"][rows]]
        geo = d["geo"][rows]
        geo = torch.cat([geo[:, :2], torch.log1p(geo[:, 2:4]), geo[:, 4:]], 1)
        obs = d["obs"][rows]
        hst = d["hist"][obs, w]                        # (R,H,2) relative to the fix
        msk = d["hmask"][obs, w].float()
        cos, sin = d["rot"][rows, 0:1], d["rot"][rows, 1:2]
        rel = hst - d["foot_off"][rows][:, None, :]    # relative to foot point on the arc
        along = rel[..., 0] * cos + rel[..., 1] * sin
        perp = -rel[..., 0] * sin + rel[..., 1] * cos
        fo = d["foot_off"][rows]
        fix_perp = (-(-fo[:, 0]) * sin[:, 0] + (-fo[:, 1]) * cos[:, 0])[:, None]   # fix offset from foot, perp
        fix_along = ((-fo[:, 0]) * cos[:, 0] + (-fo[:, 1]) * sin[:, 0])[:, None]
        hfeat = torch.cat([along * msk, perp * msk, msk, fix_along, fix_perp], 1)
        hc = self.hist(hfeat)
        wo = torch.zeros(len(rows), len(WINDOWS_S))
        wo[:, w] = 1
        return self.head(torch.cat([e, geo, hc, wo], 1)).squeeze(-1)


def to_torch(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k == "S":
            continue
        if isinstance(v, np.ndarray):
            out[k] = torch.from_numpy(v) if v.dtype != bool else torch.from_numpy(v)
    return out


def segment_log_softmax(z: torch.Tensor, obs: torch.Tensor, n_fix: int):
    mx = torch.full((n_fix,), -1e30).scatter_reduce(0, obs, z, reduce="amax", include_self=True)
    ez = torch.exp(z - mx[obs])
    s = torch.zeros(n_fix).index_add_(0, obs, ez)
    return z - mx[obs] - torch.log(s[obs])
