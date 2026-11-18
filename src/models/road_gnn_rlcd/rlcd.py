"""RLCD-inspired window selection on top of a FROZEN, calibrated Model 2 scorer.

Inspiration: Gao et al., "Beyond Outcome Rewards: Post-Rationale Distribution
Optimization for Selective Prediction in Reasoning Models", arXiv:2609.38850
(2026; code repo "openjev-rlcd"). That work trains reasoning LLMs with RL whose
reward is a proper scoring rule applied to the answer DISTRIBUTION produced after
a sampled rationale. It says nothing about GPS or road matching.

OUR ADAPTATION (not validated by that paper):
  "rationale"  -> a choice of how much GPS history the scorer sees
  action a     in {0: current fix only, 1: short 5 s window, 2: long 15 s window}
  distribution -> the frozen scorer's calibrated p_a(. | fix) over candidate roads
  reward       -> r_a = -Brier(p_a, y) = -sum_i (p_a,i - y_i)^2   (proper scoring rule;
                  y = derived point label; a candidate miss adds +1 for every action)
The reference road enters ONLY the training reward.

State s (inference-available only): seconds of history available since trip start
/ last gap, largest time step in the last 15 s, raw GPS speed, raw-heading
reliability, number of candidates, nearest distance, gap between the two nearest
candidate distances, and for each action the max probability, entropy and
whether its top candidate agrees with the other actions.

Policies (same small network, trained on cross-fitted OOF predictions):
  direct  (full information): all three rewards are observed offline, so we
          regress the per-action Brier and act argmin. Cheap and strong.
  expected (full information): maximise sum_a pi(a|s) r_a exactly. Its gradient is
          E_{a~pi}[ r_a grad log pi(a|s) ] -- i.e. the EXPECTED REINFORCE gradient.
          So it is the same objective as RL, computed without sampling.
  rlcd    (bandit / RL): sample a ~ pi(.|s), observe only r_a, update with
          REINFORCE + a learned state-value baseline (variance reduction) and a
          small entropy bonus. This is an unbiased Monte-Carlo estimate of the
          'expected' gradient; any advantage over 'expected'/'direct' must come
          from optimisation behaviour, not from new information.
Inference rule for all policies: deterministic, a* = argmax_a pi(a|s)
(direct: argmin predicted Brier).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

ACTIONS = 3
STATE_NAMES = ["hist_avail", "max_dt_15s", "raw_speed", "raw_rel", "n_cands", "d1", "d2_minus_d1"] + \
              [f"{k}_a{a}" for a in range(ACTIONS) for k in ("maxp", "entropy", "agree")]


def per_fix_dists(p_flat, ptr):
    """max prob, entropy, argmax row per fix for one action."""
    T = len(ptr) - 1
    mx = np.zeros(T); ent = np.zeros(T); arg = np.full(T, -1)
    for f in range(T):
        s, e = ptr[f], ptr[f + 1]
        if e > s:
            q = p_flat[s:e]
            mx[f] = q.max(); ent[f] = -(q * np.log(np.clip(q, 1e-12, 1))).sum(); arg[f] = s + int(np.argmax(q))
    return mx, ent, arg


def brier_per_fix(p_flat, ptr, obs, y_row):
    T = len(ptr) - 1
    yv = np.zeros(len(p_flat))
    yv[y_row[y_row >= 0]] = 1
    return np.bincount(obs, weights=(p_flat - yv) ** 2, minlength=T) + (y_row < 0)


def state_features(t, xy, ptr, dist, probs_by_action):
    from pathsnap.features_raw import raw_heading
    T = len(t)
    seg_start_t = np.empty(T)
    seg_start_t[0] = t[0]
    for i in range(1, T):
        seg_start_t[i] = t[i] if t[i] - t[i - 1] > 10 else seg_start_t[i - 1]
    dt = np.diff(t, prepend=t[0])
    lo = np.searchsorted(t, t - 15, side="left")
    maxdt = np.array([dt[lo[i]:i + 1].max() for i in range(T)])
    _, rel, speed, _ = raw_heading(xy, t)
    n = np.diff(ptr)
    d1 = np.where(n > 0, dist[np.minimum(ptr[:-1], len(dist) - 1)], 200.0)
    d2 = np.where(n > 1, dist[np.minimum(ptr[:-1] + 1, len(dist) - 1)], 200.0)
    cols = [np.minimum(t - seg_start_t, 15) / 15, np.minimum(maxdt, 10) / 10, np.minimum(speed, 40) / 40, rel,
            n / 16, np.minimum(d1, 200) / 50, np.minimum(d2 - d1, 200) / 50]
    args = []
    stats = []
    for p in probs_by_action:
        mx, ent, arg = per_fix_dists(p, ptr)
        stats.append((mx, ent))
        args.append(arg)
    for a in range(ACTIONS):
        others = [args[b] for b in range(ACTIONS) if b != a]
        agree = np.mean([args[a] == o for o in others], axis=0)
        cols += [stats[a][0], stats[a][1], agree]
    return np.column_stack(cols).astype(np.float32)


class Policy(nn.Module):
    def __init__(self, d, h=32):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.pi = nn.Linear(h, ACTIONS)
        self.v = nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)


def _norm_fit(S):
    mu, sd = S.mean(0), S.std(0) + 1e-6
    return mu, sd


def train_policy(S, R, mode, epochs=30, batch=4096, lr=2e-3, seed=0, entropy=0.01):
    """S: (N,d) states, R: (N,3) rewards (= -Brier). Returns (callable act(S)->actions, log)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    mu, sd = _norm_fit(S)
    X = torch.tensor((S - mu) / sd)
    Rt = torch.tensor(R, dtype=torch.float32)
    net = Policy(S.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    log = []
    for ep in range(epochs):
        perm = torch.randperm(len(X), generator=g)
        tot = 0.0
        for i in range(0, len(X), batch):
            b = perm[i:i + batch]
            logits, v = net(X[b])
            if mode == "direct":
                loss = ((logits - Rt[b]) ** 2).mean()           # logits regress the 3 rewards
            elif mode == "expected":
                pi = torch.softmax(logits, -1)
                loss = -(pi * Rt[b]).sum(-1).mean()
            elif mode == "rlcd":
                logp = torch.log_softmax(logits, -1)
                pi = logp.exp()
                a = torch.multinomial(pi.detach(), 1, generator=g).squeeze(-1)
                r = Rt[b, a]
                adv = (r - v).detach()
                loss = -(adv * logp[torch.arange(len(b)), a]).mean() + 0.5 * ((v - r) ** 2).mean() \
                    - entropy * (-(pi * logp).sum(-1)).mean()
            else:
                raise ValueError(mode)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        with torch.no_grad():
            lg, _ = net(X)
            act = lg.argmax(-1)
            mean_r = Rt[torch.arange(len(X)), act].mean().item()
        log.append(dict(epoch=ep + 1, loss=tot / len(X), train_mean_reward_greedy=mean_r,
                        action_freq=np.bincount(act.numpy(), minlength=ACTIONS).tolist()))

    @torch.no_grad()
    def act(Snew):
        lg, _ = net(torch.tensor((Snew - mu) / sd))
        return lg.argmax(-1).numpy()
    act.state = dict(mu=mu, sd=sd, weights={k: v.numpy() for k, v in net.state_dict().items()}, mode=mode)
    return act, log
