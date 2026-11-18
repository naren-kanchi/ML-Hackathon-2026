# HMM decoding - scorer `lgbm_ranker`, partition `train`

Trips: 80. Emission: `lgbm_ranker`. Transition: Newson & Krumm exponential (beta = 1.01 m). Parameters estimated on train only.

Values are means over trips (metrics in %, decode time in seconds per trip).

|                |   classifier |   classifier+hmm |
|:---------------|-------------:|-----------------:|
| Top-1 acc (%)  |        91.93 |            92.26 |
| Cand. miss (%) |         0.65 |             0.65 |
| Continuity (%) |        90.95 |            95.03 |
| Dir P          |        72.22 |            88.35 |
| Dir R          |        98.1  |            98.7  |
| Dir F1         |        81.97 |            92.89 |
| Dir len-P      |        77.62 |            93.16 |
| Dir len-R      |        98.92 |            98.85 |
| Undir F1       |        86.28 |            95.3  |
| Undir len-P    |        84.27 |            96.18 |
| Undir len-R    |        98.95 |            98.88 |
| Route gaps     |        23.2  |             1.65 |
| Chain breaks   |       nan    |             3.36 |
| Decode s       |         0.01 |             0.09 |

Pooled top-1 accuracy over all labelled points: classifier 93.18%, classifier+HMM 93.80%.

**Continuity check:** classifier+HMM 95.03% vs classifier-only 90.95% -> HIGHER; HMM higher on 77/80 trips.

Chain breaks: 269 total (197 from unreachable steps). Total HMM decode time 7.0 s.

Notes: sequence metrics compare the reconstructed route (shortest-path filled) with the reference route between the first and last labelled point; 'Dir' = exact directed arc, 'Undir' = arc or its reverse. Candidate miss rate is the ceiling on accuracy and is the same for both methods.