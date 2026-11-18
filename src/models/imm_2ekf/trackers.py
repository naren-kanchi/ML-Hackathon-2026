"""GPS-only motion trackers: turning EKF (variant B) and IMM straight+turn (variant C).

Input: projected GPS positions z_t (m) and timestamps t. Nothing else.
Speed, heading and turn rate are latent states INFERRED from positions.

State s = [x, y, v, psi, omega]  (position m, speed m/s, heading rad, turn rate rad/s)

Models (both use the same state so the IMM can mix them):
  CT  (coordinated turn): x += v/w (sin(psi+w dt) - sin psi), y += v/w (cos psi - cos(psi+w dt)),
      psi += w dt, w constant. For |w| < 1e-4 the straight-line limit is used.
  CV  (straight): x += v dt cos psi, y += v dt sin psi, psi const, w -> 0.
Jacobians are computed by central finite differences (handles the w -> 0 limit).

Measurement: z = [x, y] + N(0, R), R = sigma_z^2 I.
Process noise (discrete approximation, actual elapsed dt):
  q_xy = (0.5 sa dt^2)^2, q_v = (sa dt)^2, q_psi = (spsi dt)^2, q_w = (sw dt)^2.

Initialisation: two-point. Fix 0 sets the position; the first later fix >= 3 m
away sets speed and heading (variance from R / displacement). The tracker re-
initialises the same way after a gap > REINIT_GAP_S.
Low speed: when v is small the heading is unobservable; its variance grows via
q_psi and downstream features use the filter's own heading std as reliability.
Angles: every heading difference passes through wrap(); IMM mixing/combination
use circular means relative to a reference heading.

For every fix the tracker records BOTH
  prior     (predicted from fixes < t, before seeing z_t), and
  posterior (after the update with z_t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REINIT_GAP_S = 10.0
EPS_W = 1e-4


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class TrackerParams:
    sigma_z: float = 6.0        # GPS position noise std (m)        -- fitted on train (scripts/fit_tracker_noise.py)
    sa: float = 1.5             # accel noise (m/s^2)
    spsi_cv: float = 0.02       # heading noise, straight model (rad/s)
    spsi_ct: float = 0.02       # heading noise, turn model (rad/s)
    sw: float = 0.15            # turn-rate noise, turn model (rad/s^2)
    p_switch: float = 0.05      # IMM mode switch rate (per second)


def f_ct(s, dt):
    x, y, v, p, w = s
    if abs(w) > EPS_W:
        return np.array([x + v / w * (np.sin(p + w * dt) - np.sin(p)),
                         y + v / w * (np.cos(p) - np.cos(p + w * dt)), v, p + w * dt, w])
    return np.array([x + v * dt * np.cos(p), y + v * dt * np.sin(p), v, p + w * dt, w])


def f_cv(s, dt):
    x, y, v, p, w = s
    return np.array([x + v * dt * np.cos(p), y + v * dt * np.sin(p), v, p, 0.0])


def jac(f, s, dt):
    J = np.empty((5, 5))
    h = np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-6])
    for k in range(5):
        e = np.zeros(5)
        e[k] = h[k]
        d = f(s + e, dt) - f(s - e, dt)
        d[3] = wrap(d[3])
        J[:, k] = d / (2 * h[k])
    return J


def Qmat(dt, sa, spsi, sw):
    return np.diag([(0.5 * sa * dt * dt) ** 2] * 2 + [(sa * dt) ** 2, (spsi * dt) ** 2 + 1e-8, (sw * dt) ** 2 + 1e-10])


H = np.zeros((2, 5))
H[0, 0] = H[1, 1] = 1.0


class EKF:
    def __init__(self, f, sa, spsi, sw, sigma_z):
        self.f, self.sa, self.spsi, self.sw = f, sa, spsi, sw
        self.R = np.eye(2) * sigma_z ** 2
        self.s = None
        self.P = None

    def predict(self, dt, s=None, P=None):
        s = self.s if s is None else s
        P = self.P if P is None else P
        F = jac(self.f, s, dt)
        s2 = self.f(s, dt)
        s2[3] = wrap(s2[3])
        P2 = F @ P @ F.T + Qmat(dt, self.sa, self.spsi, self.sw)
        return s2, P2

    def update(self, s, P, z):
        y = z - s[:2]
        S = P[:2, :2] + self.R
        Si = np.linalg.inv(S)
        K = P[:, :2] @ Si
        s2 = s + K @ y
        P2 = (np.eye(5) - K @ H) @ P
        P2 = 0.5 * (P2 + P2.T)
        if s2[2] < 0:  # keep speed non-negative: (−v, psi) == (v, psi+pi)
            s2[2] = -s2[2]
            s2[3] += np.pi
        s2[3] = wrap(s2[3])
        lik = np.exp(-0.5 * y @ Si @ y) / (2 * np.pi * np.sqrt(np.linalg.det(S)))
        nis = float(y @ Si @ y)
        return s2, P2, lik, nis


def _init_state(z0, z1, dt, sigma_z):
    d = z1 - z0
    disp = np.hypot(*d)
    v = disp / dt
    psi = np.arctan2(d[1], d[0])
    s = np.array([z1[0], z1[1], v, psi, 0.0])
    P = np.diag([sigma_z ** 2] * 2 + [2 * sigma_z ** 2 / dt ** 2 + 1.0,
                                      min(2 * sigma_z ** 2 / max(disp, 1e-3) ** 2, np.pi ** 2), 0.05 ** 2])
    return s, P


def _circ_mix(states, w, ref):
    m = np.einsum("i,ij->j", w, states)
    m[3] = wrap(ref + np.sum(w * wrap(states[:, 3] - ref)))
    return m


def _mix_cov(states, covs, w, mean):
    P = np.zeros((5, 5))
    for i in range(len(w)):
        d = states[i] - mean
        d[3] = wrap(d[3])
        P += w[i] * (covs[i] + np.outer(d, d))
    return P


def run_tracker(xy: np.ndarray, t: np.ndarray, kind: str, p: TrackerParams = TrackerParams(),
                reinit_gap_s: float = REINIT_GAP_S):
    """kind in {'ekf','imm'}. Returns dict of per-fix arrays (prior_*, post_*, mu_turn, nis, ready)."""
    T = len(t)
    out = {k: np.zeros(T) for k in ["prior_x", "prior_y", "prior_vx", "prior_vy", "prior_vxy",
                                    "prior_psi", "prior_psi_std", "post_v", "post_psi", "post_psi_std",
                                    "post_w", "post_w_std", "mu_turn", "nis", "loglik"]}
    out["prior_cov"] = np.zeros((T, 2, 2))
    ready = np.zeros(T, bool)      # True when a prior existed for this fix
    if kind == "ekf":
        filt = [EKF(f_ct, p.sa, p.spsi_ct, p.sw, p.sigma_z)]
    else:
        filt = [EKF(f_cv, p.sa, p.spsi_cv, 1e-3, p.sigma_z), EKF(f_ct, p.sa, p.spsi_ct, p.sw, p.sigma_z)]
    M = len(filt)
    mu = np.ones(M) / M
    states = covs = None
    anchor = 0          # index of the fix used as the position-only anchor while (re)initialising
    for i in range(T):
        dt = t[i] - t[i - 1] if i else 0.0
        if states is None or dt > reinit_gap_s:
            if states is not None:  # gap re-init
                anchor = i
                states = None
                mu = np.ones(M) / M
            if i == anchor or np.hypot(*(xy[i] - xy[anchor])) < 3.0:
                if i != anchor and t[i] - t[anchor] > reinit_gap_s:
                    anchor = i
                out["mu_turn"][i] = mu[-1] if M > 1 else 1.0
                continue
            s0, P0 = _init_state(xy[anchor], xy[i], t[i] - t[anchor], p.sigma_z)
            states = np.stack([s0] * M)
            covs = np.stack([P0] * M)
            out["post_v"][i], out["post_psi"][i] = s0[2], s0[3]
            out["post_psi_std"][i] = np.sqrt(P0[3, 3])
            out["mu_turn"][i] = mu[-1] if M > 1 else 1.0
            continue
        # ---- IMM mixing (no-op for a single EKF)
        if M > 1:
            ps = 1 - np.exp(-p.p_switch * dt)
            Pi = np.array([[1 - ps, ps], [ps, 1 - ps]])
            cbar = Pi.T @ mu
            mix_w = (Pi * mu[:, None]) / np.maximum(cbar[None, :], 1e-300)   # mix_w[i,j] = mu_{i|j}
            ms, mP = [], []
            for j in range(M):
                m = _circ_mix(states, mix_w[:, j], states[j, 3])
                ms.append(m)
                mP.append(_mix_cov(states, covs, mix_w[:, j], m))
            states, covs = np.stack(ms), np.stack(mP)
        else:
            cbar = mu
        # ---- predict (prior for fix i) with actual elapsed dt
        pri = [f.predict(dt, states[j], covs[j]) for j, f in enumerate(filt)]
        pri_s = np.stack([a for a, _ in pri])
        pri_P = np.stack([b for _, b in pri])
        ref = pri_s[int(np.argmax(cbar)), 3]
        comb_s = _circ_mix(pri_s, cbar, ref)
        comb_P = _mix_cov(pri_s, pri_P, cbar, comb_s)
        out["prior_x"][i], out["prior_y"][i] = comb_s[0], comb_s[1]
        out["prior_cov"][i] = comb_P[:2, :2]
        out["prior_psi"][i] = comb_s[3]
        out["prior_psi_std"][i] = np.sqrt(max(comb_P[3, 3], 0))
        ready[i] = True
        # ---- update with z_i, mode probabilities from measurement likelihoods
        z = xy[i]
        upd = [f.update(pri_s[j], pri_P[j], z) for j, f in enumerate(filt)]
        lik = np.array([u[2] for u in upd])
        mu = cbar * lik
        tot = mu.sum()
        mu = mu / tot if tot > 0 else np.ones(M) / M
        states = np.stack([u[0] for u in upd])
        covs = np.stack([u[1] for u in upd])
        post = _circ_mix(states, mu, states[int(np.argmax(mu)), 3])
        postP = _mix_cov(states, covs, mu, post)
        out["post_v"][i] = post[2]
        out["post_psi"][i] = post[3]
        out["post_psi_std"][i] = np.sqrt(max(postP[3, 3], 0))
        out["post_w"][i] = post[4]
        out["post_w_std"][i] = np.sqrt(max(postP[4, 4], 0))
        out["mu_turn"][i] = mu[-1] if M > 1 else 1.0
        out["nis"][i] = sum(m_ * u[3] for m_, u in zip(mu, upd))
        out["loglik"][i] = np.log(max(tot, 1e-300))
    out["ready"] = ready
    return out


TRACK_NAMES = ["prior_seg_d", "prior_seg_z", "prior_cos", "post_cos", "post_sin_abs", "post_rel",
               "post_v", "post_w_abs", "turn_into", "nis", "ready"]


def tracker_features(ctx, tr: dict, imm: bool, sigma_z: float | None = None) -> np.ndarray:
    """Candidate features from tracker output.
    prior_seg_d  distance from the PRIOR predicted position to the candidate segment (m)
    prior_seg_z  that distance / sqrt(n^T P_prior n + sigma_z^2) (uncertainty-normalised)
    prior_cos    cos(prior heading - arc heading), weighted by prior heading reliability
    post_cos     cos(posterior heading - arc heading), weighted by reliability
    post_sin_abs |sin(...)| weighted by reliability
    post_rel     reliability = exp(-psi_std^2/2) * 1[v > 1.5 m/s]
    post_v, post_w_abs  posterior speed and |turn rate|
    turn_into    sign(w) * sin(arc heading - post heading) * min(|w|*10,1): candidate lies
                 in the direction the tracker is turning
    nis          normalised innovation squared of the update (same for all candidates of the fix)
    ready        1 if a prior existed (0 at trip start / just after re-init)
    (+ mu_turn for the IMM)
    sigma_z: GPS noise used in prior_seg_z. None keeps the original behaviour (class
    default 6.0 m) so the first Model 1 run stays reproducible; pass the fitted value.
    """
    c, rn = ctx.cands, ctx.rn
    o = c.obs
    a = c.arc
    pp = np.column_stack([tr["prior_x"], tr["prior_y"]])[o]
    d, _, _, foot = rn.project_points(pp, a)
    nvec = pp - foot
    nn = np.linalg.norm(nvec, axis=1, keepdims=True)
    nvec = np.where(nn > 1e-6, nvec / np.maximum(nn, 1e-6), 0.0)
    C = tr["prior_cov"][o]
    var_n = np.einsum("ni,nij,nj->n", nvec, C, nvec) + (TrackerParams.sigma_z if sigma_z is None else sigma_z) ** 2
    ready = tr["ready"][o]
    th = rn.heading[a]
    prel = np.exp(-0.5 * tr["prior_psi_std"][o] ** 2) * ready
    v = tr["post_v"][o]
    rel = np.exp(-0.5 * tr["post_psi_std"][o] ** 2) * (v > 1.5)
    dpost = wrap(th - tr["post_psi"][o])
    w = tr["post_w"][o]
    F = [np.where(ready, d, 0.0), np.where(ready, d / np.sqrt(var_n), 0.0),
         prel * np.cos(wrap(th - tr["prior_psi"][o])), rel * np.cos(dpost), rel * np.abs(np.sin(dpost)), rel,
         np.minimum(v, 60), np.abs(w), np.sign(w) * np.sin(dpost) * np.minimum(np.abs(w) * 10, 1) * rel,
         np.minimum(tr["nis"][o], 50), ready]
    if imm:
        F.append(tr["mu_turn"][o])
    return np.column_stack(F).astype(np.float32)
