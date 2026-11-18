# HMM decoding - scorer `lgbm_ranker`, partition `student_test`

Trips: 10. Emission: `lgbm_ranker`. Transition: Newson & Krumm exponential (beta = 1.01 m). Parameters estimated on train only.

Values are means over trips (metrics in %, decode time in seconds per trip).

|                |   classifier |   classifier+hmm |
|:---------------|-------------:|-----------------:|
| Top-1 acc (%)  |        92.98 |            94.69 |
| Cand. miss (%) |         0.29 |             0.29 |
| Continuity (%) |        92.29 |            95.99 |
| Dir P          |        71.75 |            91.11 |
| Dir R          |        99.3  |            99.64 |
| Dir F1         |        82.71 |            95.05 |
| Dir len-P      |        79.64 |            94.39 |
| Dir len-R      |        99.56 |            99.79 |
| Undir F1       |        86.82 |            96.69 |
| Undir len-P    |        85.85 |            96.42 |
| Undir len-R    |        99.64 |            99.79 |
| Route gaps     |        20.7  |             0.3  |
| Chain breaks   |       nan    |             0.6  |
| Decode s       |         0.01 |             0.07 |

Pooled top-1 accuracy over all labelled points: classifier 93.37%, classifier+HMM 95.04%.

**Continuity check:** classifier+HMM 95.99% vs classifier-only 92.29% -> HIGHER; HMM higher on 10/10 trips.

Chain breaks: 6 total (6 from unreachable steps). Total HMM decode time 0.7 s.

Notes: sequence metrics compare the reconstructed route (shortest-path filled) with the reference route between the first and last labelled point; 'Dir' = exact directed arc, 'Undir' = arc or its reverse. Candidate miss rate is the ceiling on accuracy and is the same for both methods.