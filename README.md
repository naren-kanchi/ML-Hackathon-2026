# ML-Hackathon-2026

## Data pipeline

The processed data and features are not stored in the repository. Rebuild them locally from the raw dataset.

**Prerequisites**

- The raw dataset must be in `Dataset/train` and `Dataset/student_test`. It is not committed to the repository.
- Python environment set up:

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

**Rebuild** (run from the repository root):

```bash
# 1. Preprocess raw trips -> data/processed/
python -m src.data.preprocess Dataset data/processed

# 2. Build features -> data/features/
python -m src.features.build_features data/processed data/features
```

Both output folders are git-ignored. Small summaries of each run (`preprocess_summary.csv`, `preprocess_config.json`, `feature_summary.csv`, `feature_config.json`) are kept in `reports/features/`.

## HistGradientBoosting candidate classifier

This implementation trains a binary HistGradientBoosting classifier that scores whether a road-arc candidate matches the supplied reference route at a GPS observation. Arc IDs are local to each trip's graph and are not treated as global classes.

### Data layout

Extract the supplied student archive so the directory passed to `--data` contains `train/` and `student_test/`, with one trip directory per partition. The evaluator-test partition is not used. Do not commit raw data.

### Train and evaluate

```powershell
python -m pip install -r requirements.txt
python src/models/histgradientboosting/train_histgradientboosting.py --data path/to/student-data --output artifacts/models
```

The command writes `histgradientboosting.joblib` and `metrics.json`. Add `--trials 5` to run Gaussian-process Bayesian optimization over learning rate, iterations, leaf count, minimum leaf samples, and L2 regularization. The objective is mean trip-grouped CV macro F1. Trial results are written into the metrics JSON.

### Recorded run

The checked-in model and metrics were produced with seed 42, 80 training trips, five-fold GroupKFold by trip, and the 10 student-test trips. Five-trial Bayesian optimization selected learning_rate=0.03846, max_iter=210, max_leaf_nodes=60, min_samples_leaf=10, and l2_regularization=9.4762. Tuned CV candidate-level macro F1 was 0.9181; student-test candidate-level macro F1 was 0.9102 (accuracy 0.9840, log loss 0.0426). Candidate miss rates were 0.11% (train) and 0.22% (student test). These are candidate classification results, not route-level accuracy. Route labels are ordered arc sequences; the implementation projects each route arc to its nearest GPS sample using the arc midpoint and records candidate misses separately.

This is one model experiment. The hackathon instructions also require at least five model comparisons, route decoding, API/UI work, and a technical report; those components are not claimed as complete here. Do not present unrun experiments as results.
