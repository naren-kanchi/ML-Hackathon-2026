# Road-graph GNN candidate scorer + RLCD-inspired window selection

The scorer produces candidate-road probabilities, which the HMM/Viterbi
decoder (`pathsnap/hmm.py`) then decodes.

## Model (`model.py`)

- **Road graph per trip.** The GNN's nodes are the canonical directed arcs
  within 2 hops of any candidate. An edge a → b exists when `to(a) == from(b)`,
  carrying the turn angle (cos/sin), a U-turn flag and the length of b. Arc
  inputs are length, two-way flag and in/out degree. There are no trip or arc
  IDs and no absolute heading.
- **Message passing (variant B, `mp`).** 2 layers. Each arc averages messages
  from its successors and its predecessors separately. Variant A (`nomp`) uses
  the same encoder with the message terms replaced by per-arc layers.
- **GPS history.** Uses only fixes from (t − W, t], never across a gap
  > 10 s, expressed in the candidate arc's own frame (along/perpendicular
  offsets, /50 m).
  - Windows W: 0 s, 5 s, 15 s. These come from training statistics: median
    moving speed 12.1 m/s, so 5 s ≈ 60 m (enough displacement for a heading
    well above 5 m GPS noise) and 15 s ≈ 180 m (spans a typical ramp
    divergence).
  - At the start of a trip, the window holds only the fixes that exist.
- **Score.** An MLP over [arc embedding, geometry, history code, window
  one-hot] gives one logit per candidate. A masked softmax over the fix's
  candidates gives one distribution per window. The model is trained on all
  3 windows jointly (cross-entropy), so every window is usable.
- **Calibration.** One temperature per window, fitted on out-of-fold
  predictions.

## RLCD-inspired window policy (`rlcd.py`, `run_rlcd.py`)

- **Inspiration:** Gao et al., *Beyond Outcome Rewards: Post-Rationale
  Distribution Optimization for Selective Prediction in Reasoning Models*,
  arXiv:2609.38850 (2026; code repo "openjev-rlcd"). That paper uses RL with a
  proper-scoring-rule reward on an LLM's answer distribution. It does not
  study GPS or map matching.
- **Our adaptation:** the action is which GPS window the frozen scorer uses
  (0 / 5 / 15 s). The reward is the negative multiclass Brier score of that
  window's calibrated distribution against the derived point label. The label
  is used only in the training reward.
- **Policy state (inference-available only):** history available, recent
  maximum time step, raw speed, heading reliability, candidate count, nearest
  distance, gap between the 2 nearest distances, and each window's max
  probability, entropy and top-1 agreement.
- **Methods compared** on identical cross-fitted folds (policy for fold k is
  trained on the other folds' out-of-fold predictions):
  1. `fixed_raw`: best single window (chosen on CV), no calibration
  2. `fixed_temp`: the same window plus temperature scaling
  3. `direct`: full-information selector (regress all 3 Brier scores, act argmin)
  4. `expected`: maximise Σₐ π(a|s)·rₐ exactly. Its gradient *is* the
     expected REINFORCE gradient, so it is the same objective as (5),
     computed without sampling.
  5. `rlcd`: REINFORCE (sample one window, observe only its reward), learned
     value baseline, entropy bonus 0.01
  6. `oracle`: best window per fix using the label. This is an upper bound
     only, not a method.
- **Inference rule:** argmax π (for `direct`, argmin of the predicted Brier).

## Results (measured; 78 CV trips / 183,260 fixes; 10 student-test trips / 19,160 fixes)

### A vs B: does the road graph help? (15 s window, chosen on CV)

| Variant | Params | CV Top-1, clf | CV Top-1, +HMM | ST Top-1, clf | ST Top-1, +HMM | CV log loss | CV trip macro F1, +HMM |
|---|---|---|---|---|---|---|---|
| A: no message passing | 14,209 | 0.8930 | 0.9172 | 0.8753 | 0.9011 | 0.4029 | 0.9001 |
| B: message passing | 18,945 | 0.8948 | 0.9179 | 0.8854 | 0.9156 | 0.3990 | 0.9017 |

Paired by fold (B − A, Top-1 points):
- classifier-only: −0.10, +0.26, +0.06, +0.85, −0.11 (B better on 51 of 78 trips)
- with HMM: −0.36, −0.03, −0.05, +0.56, +0.31 (B better on 42, worse on 33)

**Conclusion:** on CV, local road connections give at most a small
classifier gain that mostly disappears after Viterbi. The decoder already
uses connectivity. The larger student-test gap rests on only 10 trips and
should not be over-read. Parameter counts differ (14.2k vs 18.9k).

### Window choice and RLCD (scorer B, frozen)

Per-window CV out-of-fold results (calibrated):

| Window | Top-1 | Log loss | Brier |
|---|---|---|---|
| 0 s | 0.595 | 0.916 | 0.486 |
| 5 s | 0.881 | 0.430 | 0.168 |
| 15 s | 0.895 | 0.399 | 0.150 |

Selection methods (classifier-only unless marked; ECE = top-1 expected
calibration error vs derived labels, 10 bins; reliability bins with counts in
`results/rlcd_mp/reliability.csv`):

| Method | CV Brier | CV log loss | CV Top-1 | CV ECE | CV Top-1 +HMM | ST Brier | ST Top-1 +HMM |
|---|---|---|---|---|---|---|---|
| fixed_raw (15 s) | 0.1502 | 0.4003 | 0.8948 | 0.0105 | 0.9181 | 0.1619 | 0.9160 |
| fixed_temp (15 s) | 0.1500 | 0.3990 | 0.8948 | 0.0058 | 0.9179 | 0.1616 | 0.9156 |
| direct | 0.1509 | 0.4036 | 0.8944 | 0.0087 | 0.9189 | 0.1580 | 0.9163 |
| expected | 0.1493 | 0.3981 | 0.8951 | 0.0078 | 0.9176 | 0.1569 | 0.9161 |
| rlcd | 0.1487 | 0.3974 | 0.8955 | 0.0082 | 0.9181 | 0.1584 | 0.9165 |
| oracle (upper bound) | 0.1114 | 0.3410 | 0.9221 | 0.0290 | – | 0.1069 | – |

**Conclusion:**
- There is real headroom in choosing the window per fix: the oracle reaches
  Brier 0.111 versus 0.150.
- The inference-time state features capture very little of it. The best
  learned selector (`rlcd`) improves CV Brier by about 1% over a fixed
  calibrated window, and every method reaches the same Top-1 (≈ 0.918) after
  HMM decoding.
- RL gave no meaningful advantage over the full-information `expected`
  objective, which is the same objective computed without sampling.
- Temperature scaling barely changed the model (T ≈ 1.05–1.16) but halved
  ECE (0.0105 → 0.0058).

## Files

- `results/m2_{nomp,mp}/`: `cv_history.csv` (per-epoch train/val loss per
  window), `window_table.csv`, `summary.csv`, per-trip CSVs,
  `temperature.json`, `run_info.json` (timings)
- `results/rlcd_mp/`: `classifier_only.csv`, `summary_hmm.csv`,
  `reliability.csv`, `policy_training_log.csv`, per-trip CSVs
- `artifacts/model2_{nomp,mp}.pt`: final scorers refit on all 78 trips.
  Policy weights are not saved: the recommended deployment is the fixed 15 s
  window with temperature, since the policies did not change decoded accuracy.

## Reproduce (from the repo root)

```bash
python scripts/build_labels.py                       # derived labels (shared)
python -m src.models.road_gnn_rlcd.build_features_m2 # per-trip graph tensors
python -m src.models.road_gnn_rlcd.train_m2 nomp mp  # ~35 min CPU each
python -m src.models.road_gnn_rlcd.run_rlcd mp       # ~10 min incl. HMM
```

Training ran on CPU (19k-parameter model). Seeds: folds 20261118, per-fold
seed + fold index.
