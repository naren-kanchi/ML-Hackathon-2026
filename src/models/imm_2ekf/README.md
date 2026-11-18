# IMM + 2 EKF candidate scorer

GPS-only motion tracker features + a trained road-candidate classifier. The
candidate scores feed the HMM/Viterbi decoder (`pathsnap/hmm.py`).

## What the model does

1. **Tracker.** An interacting multiple model (IMM) runs two extended Kalman
   filters on the same state `[x, y, speed, heading, turn rate]`:
   - straight model (constant velocity, turn rate forced to 0)
   - coordinated-turn model (constant speed and turn rate)

   Each step it mixes the two filters' estimates, predicts with the real
   elapsed time, updates with the GPS fix, re-weights the two models by how
   well each predicted the fix, and combines them. The only inputs are GPS
   position and time. Speed, heading and turn rate are estimated, not measured.
2. **Features per candidate road.** Distance from the *predicted* position
   (before seeing the fix) to the road, normalised by the prediction's
   uncertainty; agreement between the road's direction and the tracked heading
   (both prior and posterior, weighted by heading reliability); whether the
   road lies in the direction the vehicle is turning; speed, turn rate,
   innovation size, and the turn-model probability. Code:
   `trackers.py: tracker_features`.
3. **Classifier.** A small listwise MLP (`pathsnap/listwise.py`) gives each
   candidate a logit, and a softmax over the point's candidates gives one
   probability distribution per GPS point. It is trained with cross-entropy
   on the derived route label, and one temperature is fitted on out-of-fold
   predictions.

## Parameters

| Kind | What |
|---|---|
| Learned | MLP weights, feature mean/std, calibration temperature |
| Fitted on train only (likelihood grid) | GPS noise `sigma_z`, acceleration noise `sa`, turn-rate noise `sw` |
| Fixed | straight/turn heading noise 0.02 rad/s, mode-switch rate 0.05/s, MLP 2x64, dropout 0.1, Adam 2e-3, 30 epochs, seed 20261118 |
| Restart rule | tracker re-initialises after a gap > 10 s (raw 1 Hz data) or > 30 s (team-preprocessed data, which is thinned) |

## Two runs

| Run | Data | Script | Results |
|---|---|---|---|
| `imm_1hz` | raw 1 Hz fixes, shared candidates (50 m, top 16) | `train_m1.py` (+ `build_features_m1.py`, `fit_tracker_noise.py`) | `results/imm_1hz/` |
| `imm_teamdata` | team preprocessing (`src/data/preprocess.py`, `src/features/build_features.py`) | `train_imm_teamdata.py` | `results/imm_teamdata/` |

Both use the frozen 5-fold trip-grouped split in `outputs/splits/folds.json`
(78 training trips; trips 33 and 54 are excluded because their GPS does not
follow their reference route). Student-test numbers are from one refit on all
78 trips.

## Reproduce (from the repo root)

```bash
# team-preprocessed data
python -m src.data.preprocess "Student folder" data/processed
python -m src.features.build_features data/processed data/features
python -m src.models.imm_2ekf.train_imm_teamdata

# raw 1 Hz pipeline
python scripts/build_labels.py
python -m src.models.imm_2ekf.fit_tracker_noise
python -m src.models.imm_2ekf.build_features_m1
python -m src.models.imm_2ekf.train_m1
```

## Results: team-preprocessed data, 5-fold CV (measured)

Same folds, classifier, seed and epochs for both rows. The only difference is
whether the 12 IMM features are added to the team's 17 emission features.
The denominator is team-labelled points (`true_arc >= 0`): 114,494 CV points
and 13,340 student-test points. Candidate misses count as Top-1 errors.
`cand_macro_f1` follows the HistGradientBoosting README convention (binary
candidate rows, threshold 0.5).

| Variant | Split | Obs. Top-1 | Trip-mean arc macro F1 | Obs. log loss | Cand. macro F1 | Cand. log loss | Miss rate |
|---|---|---|---|---|---|---|---|
| team features only | CV 5-fold OOF | 0.9340 | 0.8131 | 0.2488 | 0.9614 | 0.0668 | 0.32% |
| team + IMM | CV 5-fold OOF | **0.9438** | **0.8410** | **0.2084** | **0.9676** | **0.0542** | 0.32% |
| team features only | student-test | 0.9343 | 0.8193 | 0.2285 | 0.9609 | 0.0643 | 0.21% |
| team + IMM | student-test | **0.9416** | **0.8472** | **0.1943** | **0.9666** | **0.0536** | 0.21% |

- The IMM improves Top-1 in **every fold**: +1.11, +1.78, +0.77, +0.26 and +1.10 points (folds 0 to 4).
- It improves Top-1 on **9 of 10** student-test trips.
- Fold Top-1 (mean ± sd): team only 0.933 ± 0.023; team + IMM 0.943 ± 0.020.
- Fitted tracker noise (train only): `sigma_z = 5 m`, `sa = 1.0 m/s²`, `sw = 0.01 rad/s²`.
  `sw` lies on the lowest grid value, so a lower optimum is possible.
- These are classifier-only numbers (no HMM). The HMM comparison on this
  data is not run yet.
- The HistGradientBoosting numbers in the main README use different labels
  (route arc projected to the nearest GPS sample), different candidates and
  all 80 trips, so they are **not directly comparable** with this table.

Files: `results/imm_teamdata/summary.csv`, `cv_folds.csv`, `cv_history.csv`
(per-epoch train/val loss), per-trip CSVs; weights in
`artifacts/team_plus_imm_teamdata.pt` and `artifacts/team_only_teamdata.pt`;
tracker config in `configs/tracker_teamdata.json`.

## Results: raw 1 Hz pipeline (measured, `results/imm_1hz/summary.csv`)

| Split | Mode | Obs. Top-1 | Trip-mean arc macro F1 | Log loss | Continuity |
|---|---|---|---|---|---|
| CV OOF (78 trips) | classifier | 0.876 | 0.794 | 0.441 | 0.960 |
| CV OOF | + HMM | 0.914 | 0.903 | – | 0.999 |
| Student-test | classifier | 0.879 | 0.813 | 0.340 | 0.961 |
| Student-test | + HMM | 0.915 | 0.924 | – | 0.9996 |

The two runs use different points (raw 1 Hz fixes vs thinned, cleaned
points) and different candidate sets (16 vs 8 per point), so they are not
directly comparable either.
