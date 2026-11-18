# HMM decoding - scorer `gaussian`, partition `train`

Trips: 80. Emission: `gaussian` (sigma = 7.26 m). Transition: Newson & Krumm exponential (beta = 1.01 m). Parameters estimated on train only.

Values are means over trips (metrics in %, decode time in seconds per trip).

|                |   classifier |   classifier+hmm |
|:---------------|-------------:|-----------------:|
| Top-1 acc (%)  |        59.74 |            90.19 |
| Cand. miss (%) |         0.65 |             0.65 |
| Continuity (%) |        72.77 |            94.89 |
| Dir P          |        50.24 |            87.56 |
| Dir R          |        96.86 |            98.82 |
| Dir F1         |        65.08 |            92.49 |
| Dir len-P      |        54.6  |            93.11 |
| Dir len-R      |        98.13 |            99.02 |
| Undir F1       |        75.32 |            95.22 |
| Undir len-P    |        69.86 |            96.29 |
| Undir len-R    |        98.22 |            99.03 |
| Route gaps     |        80.85 |             1.64 |
| Chain breaks   |       nan    |             3.36 |
| Decode s       |         0.01 |             0.06 |

Pooled top-1 accuracy over all labelled points: classifier 59.23%, classifier+HMM 92.18%.

**Continuity check:** classifier+HMM 94.89% vs classifier-only 72.77% -> HIGHER; HMM higher on 80/80 trips.

Chain breaks: 269 total (197 from unreachable steps). Total HMM decode time 4.5 s.

Notes: sequence metrics compare the reconstructed route (shortest-path filled) with the reference route between the first and last labelled point; 'Dir' = exact directed arc, 'Undir' = arc or its reverse. Candidate miss rate is the ceiling on accuracy and is the same for both methods.