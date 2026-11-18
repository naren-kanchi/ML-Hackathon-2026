# Where HMM/Viterbi decoding helped and where it did not

Source: `per_trip.csv` from the common-set evaluation (same points, labels and
decoder for every model, default decoder settings). Per-trip numbers are in
`hmm_effect_per_trip.csv` and the summary is in `hmm_effect_summary.csv`. A
trip counts as "helped" or "hurt" when its Top-1 changes by more than 0.05
points.

| Model | CV trips helped / hurt (of 78) | CV mean gain | Student-test trips helped / hurt (of 10) | Student-test mean gain |
|---|---|---|---|---|
| LightGBM LambdaRank | 72 / 4 | +2.9 pts | 10 / 0 | +3.4 pts |
| IMM + 2 EKF | 69 / 4 | +2.1 pts | 10 / 0 | +2.6 pts |
| Road-graph GNN | 69 / 7 | +2.7 pts | 10 / 0 | +2.7 pts |
| Logistic Regression | 66 / 11 | +4.5 pts | 10 / 0 | +5.4 pts |
| HistGradientBoosting | 65 / 13 | +4.7 pts | 9 / 1 | +5.1 pts |

Decoding improves almost every trip. The weaker classifiers gain the most,
because connectivity corrects more of their errors.

## Failure case: training trip 00000079

Decoding lowered Top-1 by about 10 points on this trip for all three strong
models (LightGBM 0.928 → 0.833; IMM 0.931 → 0.833; GNN 0.918 → 0.783).

- The vehicle is on a motorway at about 37 m/s.
- At the wrong points, the classifier is confident and correct: for example,
  p = 0.98 for the true carriageway. Viterbi instead picks a parallel road 25 m
  away.
- 4 of the trip's 537 true transitions are infeasible under the decoder's own
  rules (`pathsnap.hmm.pair_route_distance`):

| Point | What happens | Decoder limit |
|---|---|---|
| 87 | GPS jumps 180.6 m in 1 s on the same arc (about 650 km/h) | 105 m (v_max · dt + slack) |
| 92 | GPS jumps 133.4 m in 1 s | 105 m |
| 382, 485 | the fix moves 15–19 m *backwards* along the same arc | 15 m backward tolerance |

- The team's spike filter removes a point only when it jumps away *and*
  straight back, so these one-way jumps survive preprocessing.
- When the true transition is infeasible, the chain restarts ("disconnected")
  or follows a connected but wrong parallel road until the evidence pulls it
  back.

**Cause:** GPS outliers that break the physical limits the decoder assumes. The
classifier is not at fault.

**Possible fixes (not yet applied):**
- remove one-way position jumps above a speed threshold during preprocessing,
- or let the transition model accept a rare "teleport" with a large penalty,
  instead of a hard infeasibility.
