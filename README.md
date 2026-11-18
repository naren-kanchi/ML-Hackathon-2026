# PathSnap: map-matching and route reconstruction (ML-Hackathon-2026)

PathSnap matches noisy GPS tracks to a directed road network. For every GPS
point, a trained model scores the nearby candidate roads. An HMM/Viterbi decoder
then picks a connected sequence of roads for the whole trip. Prediction reads
only `.track`, `.nodes` and `.arcs`; the reference `.route` is used only for
training labels and evaluation.

## Repository layout

| Path | Contents |
|---|---|
| `src/data/preprocess.py` | Cleaning: per-trip UTM projection, duplicate arcs, gap split, stationary collapse, spike removal, thinning, route alignment (labels) |
| `src/features/build_features.py` | Candidate generation (50 m, up to 8) and emission/transition features |
| `src/models/` | The five models (one folder or file each, see below) |
| `src/optimization/bayes_opt_lgbm.py` | Bayesian hyperparameter optimisation of the selected model |
| `src/hmm/`, `src/evaluation/` | The team's HMM/Viterbi decoder and metrics used by the LightGBM run |
| `pathsnap/` | Shared map-matching library: road network + spatial index, label alignment, candidates, HMM/Viterbi decoder, metrics, evaluation harness, listwise scorer |
| `scripts/` | Data inventory, label alignment, frozen folds, shared decoding of every model, common-set comparison |
| `backend/`, `frontend/` | FastAPI service (laptop 1) and React evaluator UI (laptop 2) |
| `experiments/results/` | Shared-decoder results, five-model comparison, Bayesian optimisation trials |
| `reports/` | EDA, geographic summaries and maps, feature summaries, per-model reports |
| `outputs/splits/folds.json` | Frozen 5-fold trip-grouped split used by every model |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Linux/macOS
pip install -r requirements.txt
```

Put the raw data in `Dataset/` (or `Student folder/`, or set `PATHSNAP_DATA`)
with `train/` and `student_test/` inside. Raw data, processed data, feature
tables, caches and predictions are git-ignored and must be rebuilt locally.

## Data pipeline

```bash
# shared derived labels + frozen folds (folds.json is already committed; do not regenerate)
python scripts/eda.py                       # inventory -> outputs/eda/
python scripts/build_labels.py              # monotone route-to-GPS alignment -> outputs/labels/

# team preprocessing and features
python -m src.data.preprocess Dataset data/processed
python -m src.features.build_features data/processed data/features
```

Small run summaries are kept in `reports/features/`.

**Data rules followed**
- Folds are grouped by trip and frozen: 78 training trips in 5 folds.
  Trips 33 and 54 are excluded because their GPS track does not follow
  their own reference route.
- All fitting, tuning and calibration use training trips only. The 10
  student-test trips are used only for the development comparison.
- The evaluator partition is never used here.

## The five models

| # | Model | Family | Code | Owner |
|---|---|---|---|---|
| 1 | Logistic Regression | Linear | `src/models/logistic_regression/` | team |
| 2 | HistGradientBoosting | Boosting (binary classifier) | `src/models/histgradientboosting/` | team |
| 3 | LightGBM LambdaRank | Boosting (learning to rank) | `src/models/lgbm_ranker.py` | team |
| 4 | IMM + 2 EKF + listwise MLP | Motion tracking + neural network | `src/models/imm_2ekf/` | team |
| 5 | Road-graph GNN (+ RLCD window policy) | Graph neural network | `src/models/road_gnn_rlcd/` | team |

Each model folder has its own README covering configuration, ablations and
reproduction commands. Ablations, such as IMM features removed or GNN message
passing switched off, are experiments inside models 4 and 5, not extra models.

## Five-model comparison (same points, same labels, same decoder)

Each model's own report uses its own points and labels, so those numbers are
not comparable with each other. `scripts/common_eval.py` therefore scores every
model on one common set:

- **Points and labels:** the team-preprocessed GPS points that have a reference
  label (114,494 CV points, 13,340 student-test points), using the team's labels.
- **Decoder:** the same HMM/Viterbi decoder (`pathsnap/hmm.py`, default settings).
- **Top-1:** the share of points whose chosen road equals the reference road.
  Points whose reference road is not a candidate count as wrong.
- **Macro F1:** per-trip road-class macro F1, averaged over trips. Arc IDs are
  never pooled across trips.

| Model | CV Top-1, classifier | CV Top-1, + Viterbi | CV macro F1, + Viterbi | Student-test Top-1, + Viterbi | Student-test macro F1, + Viterbi |
|---|---|---|---|---|---|
| 3. LightGBM LambdaRank | 0.934 | **0.964** | 0.905 | 0.968 | **0.925** |
| 4. IMM + 2 EKF | **0.944** | 0.963 | 0.904 | 0.968 | 0.918 |
| 5. Road-graph GNN | 0.941 | 0.961 | 0.904 | 0.954 | 0.897 |
| 1. Logistic Regression | 0.713 | 0.744 | 0.661 | 0.728 | 0.657 |
| 2. HistGradientBoosting | 0.708 | 0.742 | 0.652 | 0.724 | 0.645 |

- Viterbi decoding improves every model by 2.5 to 4 Top-1 points.
- After decoding, LightGBM and IMM + 2 EKF are within one fold standard
  deviation of each other (about ±0.015). IMM + 2 EKF is the best classifier
  before decoding, ahead in every fold.
- Models 1 and 2 are limited mainly by their candidate search. It looks for
  roads by arc **midpoint** within 80 m, so the reference road is missing for
  about 15% of points.

Full table with per-fold values and trips won:
`experiments/results/common_eval/summary.csv`. Per-model decoded results:
`experiments/results/viterbi_shared/`.

## Selected model and Bayesian optimisation

LightGBM LambdaRank was selected: it has the best CV Top-1 after decoding and
trains quickly enough to search. `src/optimization/bayes_opt_lgbm.py` runs
Optuna TPE on the 5 frozen folds:

- **Stage A:** 15 trials over the ranker's hyperparameters, maximising
  out-of-fold Top-1 before decoding.
- **Stage B:** 6 trials over the decoder's parameters, on the tuned model's
  out-of-fold probabilities, maximising out-of-fold Top-1 after Viterbi.

| LightGBM | CV Top-1 (clf) | CV Top-1 (+Viterbi) | ST Top-1 (+Viterbi) | ST macro F1 (+Viterbi) | Log loss (ST) | Inference |
|---|---|---|---|---|---|---|
| Untuned | 0.9343 | 0.9639 | 0.9681 | 0.9253 | 0.1934 | 4.4 µs/row |
| **Tuned** | **0.9368** | **0.9682** | **0.9729** | **0.9311** | **0.1881** | 6.3 µs/row |

- **Best ranker:** 359 trees, learning rate 0.027, 26 leaves, 88 minimum samples
  per leaf, L2 regularisation 10.2.
- **Best decoder:** beta 1.51 m, emission weight 2.31, v_max 54.9 m/s,
  gap break 94 s.
- Trials, the best values and both models are in
  `experiments/results/bayes_opt_lgbm/`. The search was time-boxed, so 15 + 6
  trials is a small budget.

```bash
python -m src.optimization.bayes_opt_lgbm --trials-a 15 --trials-b 6
```

## Reproducing the comparison

```bash
python -m src.models.imm_2ekf.train_imm_teamdata          # model 4 (team data), 5-fold CV
python -m src.models.road_gnn_rlcd.build_features_m2      # model 5
python -m src.models.road_gnn_rlcd.train_m2 nomp mp
python scripts/viterbi_others.py hgb logreg lgbm imm_teamdata   # shared decoder for models 1-4
python scripts/common_eval.py                              # five-model common-set table
```

## HistGradientBoosting: the team's original run

```powershell
python src/models/histgradientboosting/train_histgradientboosting.py --data path/to/student-data --output artifacts/models
```

The committed model used seed 42, all 80 training trips, five-fold GroupKFold
and 5 Gaussian-process Bayesian optimisation trials. Its selected parameters
were learning_rate 0.03846, max_iter 210, max_leaf_nodes 60, min_samples_leaf
10 and l2_regularization 9.4762.

Its own report gives candidate-level macro F1 of 0.918 (CV) and 0.910
(student-test). Those labels place each route arc at its nearest GPS sample by
arc midpoint, so they cover only points near arc midpoints. They are not the
observation-level Top-1 used in the comparison above.

## Backend API (laptop 1)

The FastAPI backend reads trip files from a server-side data directory.
Prediction uses only `.track`, `.nodes` and `.arcs`. A `.route` file is read
only after inference, to compute aggregate metrics, and is never returned.

```powershell
python -m pip install -r requirements.txt
$env:PATHSNAP_DATA_DIR = "C:\path\to\Student folder"
$env:PATHSNAP_PARTITION = "student_test"
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

- API docs are at `http://localhost:8000/docs`.
- `GET /api/v1/trips` lists the available trip IDs.
- `GET /api/v1/trips/{trip_id}` returns classifier-only and Viterbi arc
  predictions, GPS points, route geometry, continuity and aggregate metrics.

The backend currently serves the HistGradientBoosting artifact. It still has to
be switched to the selected tuned LightGBM model (see *Status*).

## React evaluator UI (laptop 2)

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL = "http://<api-laptop-ip>:8000"
npm run dev -- --host 0.0.0.0
```

- On the API laptop, set `PATHSNAP_CORS_ORIGINS` to the UI laptop's origin,
  for example `http://192.168.1.20:5173`.
- For the evaluation, set `PATHSNAP_DATA_DIR` to the server-only evaluator
  data and `PATHSNAP_PARTITION=evaluator_test`. Reference routes stay on the
  API laptop.

## Status

Done:
- EDA and data-quality notes
- Preprocessing and features
- Five models trained, cross-validated and decoded
- Common-set comparison
- Bayesian optimisation of the selected model
- FastAPI backend and React UI scaffolding

Remaining:
- Serve the tuned LightGBM model and tuned decoder in the backend
- Folium maps of predicted vs reference routes
- Report figures and the consolidated report table (log loss and inference time for every model)
- End-to-end test on two laptops
- Technical report (PDF)
