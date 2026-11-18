# Model 5 - LightGBM LambdaRank candidate ranker

Family: **Boosting - learning to rank (LambdaRank)**. Untuned configuration (tuning comes in Section 7): `n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=42`, no early stopping. Features: the 17 `EMISSION_FEATURES`.

Folds: shared team folds `outputs/splits/folds.json` (5 trip-grouped folds, seed 20261118), 78 train trips. Trips 00000033, 00000054 are listed there as label-unreliable and are excluded from training and CV; they are scored out-of-sample by the final model. The final model is trained on the same 78 trips. student_test (10 trips) is used for evaluation only.

## From ranking scores to probabilities

LambdaRank outputs an unbounded relevance score that is only meaningful *within* one GPS point's candidate group, so it cannot be read as a probability directly. Because exactly one candidate per point is the true arc, we turn the scores of each group into a distribution with a softmax, p = softmax(score / T), which sums to 1 over the point's candidates and keeps the ranking unchanged. The temperature T sets how peaked the distribution is; it is fitted on train out-of-fold scores only by minimising the multiclass log loss (bounded scalar search in [0.05, 20]), giving **T = 0.9488**. For the CV estimates each fold uses a temperature fitted on the other four folds, so no fold's probabilities are calibrated on itself; student_test never influences T.

## Candidate and observation level

| Metric | CV (mean ± std over 5 folds) | student_test |
|---|---:|---:|
| Observation Top-1 accuracy (%) | 93.29 ± 2.24 | 93.37 |
| Candidate miss rate (%) | 0.32 ± 0.31 | 0.21 |
| Log loss (candidate, binary) | 0.0657 ± 0.0201 | 0.0617 |
| Softmax NLL per point (multiclass) | 0.2000 ± 0.0611 | 0.1916 |
| ROC-AUC | 0.9941 ± 0.0029 | 0.9950 |

Two decision rules on the candidate rows (positive class = true arc; values in %):

| Rule | Metric | CV (mean ± std) | student_test |
|---|---|---:|---:|
| a) Top-1 (main rule) | Accuracy | 97.62 ± 0.84 | 97.64 |
| a) Top-1 (main rule) | Precision | 93.29 ± 2.24 | 93.37 |
| a) Top-1 (main rule) | Recall | 93.58 ± 2.11 | 93.56 |
| a) Top-1 (main rule) | F1 | 93.43 ± 2.17 | 93.46 |
| b) Threshold p ≥ 0.5 | Accuracy | 97.69 ± 0.81 | 97.67 |
| b) Threshold p ≥ 0.5 | Precision | 94.25 ± 1.97 | 94.34 |
| b) Threshold p ≥ 0.5 | Recall | 92.91 ± 2.29 | 92.62 |
| b) Threshold p ≥ 0.5 | F1 | 93.57 ± 2.13 | 93.47 |

Confusion matrices (student_test): `confusion_matrix_top1.png`, `confusion_matrix_thr05.png`. Reliability plot: `calibration.png`. Feature importance: `feature_importance_gain.png`. Per-fold numbers: `cv_folds.csv`.

Observation Top-1 counts every labelled GPS point, including points with no candidates or whose true arc is not a candidate (these are always wrong), so it is capped at 100 − candidate miss rate.

## Route level: classifier-only vs classifier + HMM

Decoder: `src.hmm.run_decoder` with this model's probabilities as emissions (emission_logp = log(clip(p, 1e-9, 1))) and the same Newson & Krumm transition (beta estimated on train) as the Gaussian run. Train = out-of-fold scores on the 78 CV trips. Values are means over trips, in %.

| Partition | Method | Top-1 | Continuity | Dir P | Dir R | Dir F1 | Undir P | Undir R | Undir F1 | Dir len-P | Dir len-R | Route gaps | Chain breaks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| student_test | LambdaRank only | 92.98 | 92.29 | 71.75 | 99.30 | 82.71 | 77.96 | 99.37 | 86.82 | 79.64 | 99.56 | 20.70 | – |
| student_test | LambdaRank + HMM | 94.69 | 95.99 | 91.11 | 99.64 | 95.05 | 94.09 | 99.64 | 96.69 | 94.39 | 99.79 | 0.30 | 0.60 |
| train (OOF) | LambdaRank only | 92.31 | 91.16 | 72.78 | 98.11 | 82.38 | 79.30 | 98.20 | 86.67 | 78.06 | 98.93 | 22.67 | – |
| train (OOF) | LambdaRank + HMM | 92.68 | 95.03 | 88.65 | 98.67 | 93.05 | 92.69 | 98.70 | 95.41 | 93.37 | 98.82 | 1.65 | 3.23 |
| student_test | *Gaussian + HMM (reference)* | 92.52 | 95.90 | 90.15 | 99.61 | 94.49 | 93.75 | 99.61 | 96.49 | 93.97 | 99.79 | 0.30 | 0.60 |

## Cost

| Item | Value |
|---|---:|
| Training time, fold 0 (s) | 4.43 |
| Training time, fold 1 (s) | 4.97 |
| Training time, fold 2 (s) | 4.38 |
| Training time, fold 3 (s) | 4.59 |
| Training time, fold 4 (s) | 4.55 |
| Training time, final fit (s) | 6.23 |
| Inference, student_test total (s) | 0.350 |
| Inference per row (µs) | 4.73 |
| Inference per trip (ms) | 35.0 |
| Trees | 500 |
| Total leaves | 15,500 |
| Model file size (MB) | 1.75 |

Inference time covers LightGBM scoring plus the group softmax for all 73,985 student_test candidate rows (feature building and decoding excluded).

## Training data notes

Points whose true arc is not among their candidates carry no ranking signal and were dropped from training only: [172, 337, 331, 323, 281] groups per CV fold, 361 in the final fit (final fit trained on 114,133 groups / 631,555 rows). They remain in every evaluation, where they count as wrong.
