"""
PathSnap - HMM emission / transition scores.

Transition  Newson & Krumm (2009) exponential on |route distance - straight-line distance|:
                logp = -abs_diff_m / beta - log(beta);   unreachable -> -inf
Emission    Gaussian placeholder on point-to-arc distance (testing / baseline):
                logp = -0.5 * (d / sigma)^2 - log(sqrt(2*pi) * sigma)
            or any classifier's probability p:  logp = log(clip(p, 1e-9, 1))

sigma and beta are estimated from TRAIN trips only (estimate_params).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.data.preprocess import _read

P_FLOOR = 1e-9


def transition_logp(transition_df: pd.DataFrame, beta: float) -> np.ndarray:
    d = transition_df["abs_diff_m"].to_numpy(float)
    lp = -d / beta - np.log(beta)
    bad = (transition_df["unreachable"].to_numpy() == 1) | ~np.isfinite(d)
    return np.where(bad, -np.inf, lp)


def gaussian_emission_logp(emission_df: pd.DataFrame, sigma: float) -> np.ndarray:
    d = emission_df["dist_m"].to_numpy(float)
    return -0.5 * (d / sigma) ** 2 - np.log(np.sqrt(2 * np.pi) * sigma)


def proba_to_logp(p) -> np.ndarray:
    """Any classifier's P(candidate is the true arc) -> emission_logp."""
    return np.log(np.clip(np.asarray(p, float), P_FLOOR, 1.0))


def _files(features_dir: str, part: str, kind: str):
    d = os.path.join(features_dir, part)
    return [os.path.join(d, f.split(".")[0]) for f in sorted(os.listdir(d)) if f"_{kind}." in f]


def estimate_params(features_dir: str = "data/features", part: str = "train") -> dict:
    """sigma = 1.4826 * median distance of true candidates (robust std),
    beta  = median |route - straight| distance of true transitions / ln 2
    (median of an exponential is beta * ln 2). TRAIN partition only."""
    if part != "train":
        raise ValueError("HMM parameters must be estimated on the train partition only")
    dists = [_read(f)[lambda x: x.label == 1].dist_m.to_numpy() for f in _files(features_dir, part, "emission")]
    diffs = [_read(f)[lambda x: (x.label == 1) & (x.unreachable == 0)].abs_diff_m.to_numpy()
             for f in _files(features_dir, part, "transition")]
    dists, diffs = np.concatenate(dists), np.concatenate(diffs)
    return {
        "sigma_m": float(1.4826 * np.median(dists)),
        "beta_m": float(np.median(diffs) / np.log(2)),
        "n_true_candidates": int(len(dists)),
        "n_true_transitions": int(len(diffs)),
        "source_part": part,
    }
