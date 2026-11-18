# HMM decoding - scorer `gaussian`, partition `student_test`

Trips: 10. Emission: `gaussian` (sigma = 7.26 m). Transition: Newson & Krumm exponential (beta = 1.01 m). Parameters estimated on train only.

Values are means over trips (metrics in %, decode time in seconds per trip).

|                |   classifier |   classifier+hmm |
|:---------------|-------------:|-----------------:|
| Top-1 acc (%)  |        59.51 |            92.52 |
| Cand. miss (%) |         0.29 |             0.29 |
| Continuity (%) |        76.15 |            95.9  |
| Dir P          |        47.12 |            90.15 |
| Dir R          |        98.85 |            99.61 |
| Dir F1         |        63.13 |            94.49 |
| Dir len-P      |        53.59 |            93.97 |
| Dir len-R      |        99.36 |            99.79 |
| Undir F1       |        73.73 |            96.49 |
| Undir len-P    |        70.71 |            96.37 |
| Undir len-R    |        99.44 |            99.79 |
| Route gaps     |        43.9  |             0.3  |
| Chain breaks   |       nan    |             0.6  |
| Decode s       |         0.01 |             0.05 |

Pooled top-1 accuracy over all labelled points: classifier 59.44%, classifier+HMM 93.46%.

**Continuity check:** classifier+HMM 95.90% vs classifier-only 76.15% -> HIGHER; HMM higher on 10/10 trips.

Chain breaks: 6 total (6 from unreachable steps). Total HMM decode time 0.5 s.

Notes: sequence metrics compare the reconstructed route (shortest-path filled) with the reference route between the first and last labelled point; 'Dir' = exact directed arc, 'Undir' = arc or its reverse. Candidate miss rate is the ceiling on accuracy and is the same for both methods.