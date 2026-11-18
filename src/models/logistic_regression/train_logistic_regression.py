
"""Logistic Regression model for PathSnap road matching."""

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import numpy as np

from scipy.spatial import cKDTree

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    log_loss,
)

# Features used to describe each GPS-to-road candidate pair
FEATURES = [
    "distance_m",
    "heading_error_rad",
    "gps_speed_mps",
    "gps_bearing_sin",
    "gps_bearing_cos",
    "segment_length_m",
    "segment_bearing_sin",
    "segment_bearing_cos",
    "candidate_degree_from",
    "candidate_degree_to",
    "endpoint_min_distance_m",
    "endpoint_max_distance_m",
    "time_delta_s",
]

LAT_M = 111195.0

def read_trip(path):
    """Read all four files belonging to one trip."""

    trip_id = path.name

    track = np.loadtxt(
        path / f"{trip_id}.track",
        dtype=float,
        ndmin=2
    )

    nodes = np.loadtxt(
        path / f"{trip_id}.nodes",
        dtype=float,
        ndmin=2
    )

    arcs = np.loadtxt(
        path / f"{trip_id}.arcs",
        dtype=int,
        ndmin=2
    )

    route_path = path / f"{trip_id}.route"

    route = (
        np.loadtxt(route_path, dtype=int, ndmin=1).ravel()
        if route_path.exists()
        else np.empty(0, dtype=int)
    )

    return track, nodes, arcs, route


def xy(lonlat, lat0):
    """Convert longitude and latitude into approximate metre coordinates."""

    return np.column_stack((
        lonlat[:, 0] * LAT_M * math.cos(math.radians(lat0)),
        lonlat[:, 1] * LAT_M
    ))

def features_for_trip(path, radius_m=80.0, max_candidates=40):
    """Generate candidate-road features and binary labels for one trip."""

    track, nodes, arcs, route = read_trip(path)

    if len(track) < 2 or len(arcs) == 0:
        return (
            np.empty((0, len(FEATURES))),
            np.empty(0, dtype=int),
            {
                "trip": path.name,
                "gps": len(track),
                "route": len(route),
                "rows": 0,
                "miss_rate": 1.0
            }
        )

    # Convert GPS and road-node coordinates into metres
    lat0 = float(np.nanmedian(track[:, 1]))

    gps = xy(track[:, :2], lat0)
    nxy = xy(nodes, lat0)

    # Keep only arcs that refer to valid road nodes
    valid = (
        (arcs[:, 0] >= 0) &
        (arcs[:, 1] >= 0) &
        (arcs[:, 0] < len(nodes)) &
        (arcs[:, 1] < len(nodes))
    )

    arc_ids = np.flatnonzero(valid)
    ends = arcs[valid]

    p0 = nxy[ends[:, 0]]
    p1 = nxy[ends[:, 1]]

    # Calculate road-segment properties
    vec = p1 - p0
    seglen = np.maximum(np.linalg.norm(vec, axis=1), 0.1)

    bearing = np.arctan2(vec[:, 1], vec[:, 0])
    mid = (p0 + p1) / 2

    # Calculate road-node degrees
    degree = np.bincount(
        ends.ravel(),
        minlength=len(nodes)
    )

    # Calculate GPS movement information
    ts = track[:, 2]

    dt = np.diff(ts, prepend=ts[0])
    dt[~np.isfinite(dt) | (dt <= 0)] = 1

    motion = np.diff(gps, axis=0, prepend=gps[:1])

    gb = np.arctan2(motion[:, 1], motion[:, 0])
    speed = np.linalg.norm(motion, axis=1) / dt

    # Build spatial search trees
    gps_tree = cKDTree(gps)
    mid_tree = cKDTree(mid)

    # Map reference-route arcs to their nearest GPS observations
    labels_at = {}

    for aid in route:

        if 0 <= aid < len(arcs) and valid[aid]:

            j = int(np.searchsorted(arc_ids, aid))

            d, oi0 = gps_tree.query(mid[j])
            oi = int(oi0)

            if oi not in labels_at or float(d) < labels_at[oi][0]:
                labels_at[oi] = (float(d), int(aid))

    rows = []
    ys = []
    missed = 0

    # Generate candidate roads for every labelled GPS observation
    for oi, (_, aid) in labels_at.items():

        nearby = mid_tree.query_ball_point(
            gps[oi],
            radius_m
        )

        c = np.asarray(nearby, dtype=int)

        if not len(c):
            c = np.array([int(mid_tree.query(gps[oi])[1])])

        dist = np.linalg.norm(
            mid[c] - gps[oi],
            axis=1
        )

        # Limit the number of candidates
        if len(c) > max_candidates:
            keep = np.argsort(dist)[:max_candidates]
            c = c[keep]
            dist = dist[keep]

        # Check whether the correct road is among the candidates
        if aid not in arc_ids[c]:
            missed += 1
            continue

        # Calculate heading difference
        delta = np.arctan2(
            np.sin(bearing[c] - gb[oi]),
            np.cos(bearing[c] - gb[oi])
        )

        # Calculate distances to road endpoints
        e0 = np.linalg.norm(p0[c] - gps[oi], axis=1)
        e1 = np.linalg.norm(p1[c] - gps[oi], axis=1)

        # Create the 13 input features
        candidate_features = np.column_stack((
            dist,
            abs(delta),
            np.full(len(c), speed[oi]),
            np.full(len(c), np.sin(gb[oi])),
            np.full(len(c), np.cos(gb[oi])),
            seglen[c],
            np.sin(bearing[c]),
            np.cos(bearing[c]),
            degree[ends[c, 0]],
            degree[ends[c, 1]],
            np.minimum(e0, e1),
            np.maximum(e0, e1),
            np.full(len(c), dt[oi])
        ))

        rows.extend(candidate_features)

        # 1 = correct road, 0 = incorrect road
        ys.extend((arc_ids[c] == aid).astype(int).tolist())

    # Store information about this trip
    meta = {
        "trip": path.name,
        "gps": int(len(track)),
        "route": int(len(route)),
        "labelled_observations": int(len(labels_at)),
        "rows": len(ys),
        "candidate_misses": int(missed),
        "miss_rate": float(missed / max(len(labels_at), 1)),
        "median_interval_s": (
            float(np.median(np.diff(ts)))
            if len(ts) > 1 else None
        )
    }

    return (
        np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURES)),
        np.asarray(ys, dtype=int),
        meta
    )

def features_for_trip(path, radius_m=80.0, max_candidates=40):
    """Generate candidate-road features and binary labels for one trip."""

    track, nodes, arcs, route = read_trip(path)

    if len(track) < 2 or len(arcs) == 0:
        return (
            np.empty((0, len(FEATURES))),
            np.empty(0, dtype=int),
            {
                "trip": path.name,
                "gps": len(track),
                "route": len(route),
                "rows": 0,
                "miss_rate": 1.0
            }
        )

    # Convert GPS and road-node coordinates into metres
    lat0 = float(np.nanmedian(track[:, 1]))

    gps = xy(track[:, :2], lat0)
    nxy = xy(nodes, lat0)

    # Keep only arcs that refer to valid road nodes
    valid = (
        (arcs[:, 0] >= 0) &
        (arcs[:, 1] >= 0) &
        (arcs[:, 0] < len(nodes)) &
        (arcs[:, 1] < len(nodes))
    )

    arc_ids = np.flatnonzero(valid)
    ends = arcs[valid]

    p0 = nxy[ends[:, 0]]
    p1 = nxy[ends[:, 1]]

    # Calculate road-segment properties
    vec = p1 - p0
    seglen = np.maximum(np.linalg.norm(vec, axis=1), 0.1)

    bearing = np.arctan2(vec[:, 1], vec[:, 0])
    mid = (p0 + p1) / 2

    # Calculate road-node degrees
    degree = np.bincount(
        ends.ravel(),
        minlength=len(nodes)
    )

    # Calculate GPS movement information
    ts = track[:, 2]

    dt = np.diff(ts, prepend=ts[0])
    dt[~np.isfinite(dt) | (dt <= 0)] = 1

    motion = np.diff(gps, axis=0, prepend=gps[:1])

    gb = np.arctan2(motion[:, 1], motion[:, 0])
    speed = np.linalg.norm(motion, axis=1) / dt

    # Build spatial search trees
    gps_tree = cKDTree(gps)
    mid_tree = cKDTree(mid)

    # Map reference-route arcs to their nearest GPS observations
    labels_at = {}

    for aid in route:

        if 0 <= aid < len(arcs) and valid[aid]:

            j = int(np.searchsorted(arc_ids, aid))

            d, oi0 = gps_tree.query(mid[j])
            oi = int(oi0)

            if oi not in labels_at or float(d) < labels_at[oi][0]:
                labels_at[oi] = (float(d), int(aid))

    rows = []
    ys = []
    missed = 0

    # Generate candidate roads for every labelled GPS observation
    for oi, (_, aid) in labels_at.items():

        nearby = mid_tree.query_ball_point(
            gps[oi],
            radius_m
        )

        c = np.asarray(nearby, dtype=int)

        if not len(c):
            c = np.array([int(mid_tree.query(gps[oi])[1])])

        dist = np.linalg.norm(
            mid[c] - gps[oi],
            axis=1
        )

        # Limit the number of candidates
        if len(c) > max_candidates:
            keep = np.argsort(dist)[:max_candidates]
            c = c[keep]
            dist = dist[keep]

        # Check whether the correct road is among the candidates
        if aid not in arc_ids[c]:
            missed += 1
            continue

        # Calculate heading difference
        delta = np.arctan2(
            np.sin(bearing[c] - gb[oi]),
            np.cos(bearing[c] - gb[oi])
        )

        # Calculate distances to road endpoints
        e0 = np.linalg.norm(p0[c] - gps[oi], axis=1)
        e1 = np.linalg.norm(p1[c] - gps[oi], axis=1)

        # Create the 13 input features
        candidate_features = np.column_stack((
            dist,
            abs(delta),
            np.full(len(c), speed[oi]),
            np.full(len(c), np.sin(gb[oi])),
            np.full(len(c), np.cos(gb[oi])),
            seglen[c],
            np.sin(bearing[c]),
            np.cos(bearing[c]),
            degree[ends[c, 0]],
            degree[ends[c, 1]],
            np.minimum(e0, e1),
            np.maximum(e0, e1),
            np.full(len(c), dt[oi])
        ))

        rows.extend(candidate_features)

        # 1 = correct road, 0 = incorrect road
        ys.extend((arc_ids[c] == aid).astype(int).tolist())

    # Store information about this trip
    meta = {
        "trip": path.name,
        "gps": int(len(track)),
        "route": int(len(route)),
        "labelled_observations": int(len(labels_at)),
        "rows": len(ys),
        "candidate_misses": int(missed),
        "miss_rate": float(missed / max(len(labels_at), 1)),
        "median_interval_s": (
            float(np.median(np.diff(ts)))
            if len(ts) > 1 else None
        )
    }

    return (
        np.asarray(rows, dtype=np.float32).reshape(-1, len(FEATURES)),
        np.asarray(ys, dtype=int),
        meta
    )

def collect(partition):
    """Collect features and labels from every trip in a partition."""

    xs = []
    ys = []
    groups = []
    info = []

    # Visit every trip folder in sorted order
    for trip in sorted(p for p in partition.iterdir() if p.is_dir()):

        # Generate features and labels for this trip
        x, y, meta = features_for_trip(trip)

        info.append(meta)

        # Add usable trip data
        if len(y):
            xs.append(x)
            ys.append(y)

            # Remember which trip each row belongs to
            groups.extend([trip.name] * len(y))

    # Stop if no usable training data exists
    if not xs:
        raise RuntimeError(f"No usable labels in {partition}")

    # Combine all trips into single arrays
    return (
        np.vstack(xs),
        np.concatenate(ys),
        np.asarray(groups),
        info
    )

def build_model(C=1.0):
    """Create a scaled Logistic Regression pipeline."""

    model = Pipeline([
        ("scaler", StandardScaler()),

        ("classifier", LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        ))
    ])

    return model

def cross_validate_model(x, y, groups):
    """Evaluate Logistic Regression using GroupKFold."""

    n_splits = min(5, len(np.unique(groups)))

    if n_splits < 2:
        raise ValueError("At least 2 training trips are required.")

    cv = GroupKFold(n_splits=n_splits)

    folds = []

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(x, y, groups), start=1
    ):

        model = build_model()

        start = time.perf_counter()

        model.fit(x[train_idx], y[train_idx])

        fit_seconds = time.perf_counter() - start

        predictions = model.predict(x[val_idx])

        probabilities = model.predict_proba(x[val_idx])[:, 1]

        metrics = {
            "fold": fold,
            "validation_trips": sorted(set(groups[val_idx])),
            "accuracy": float(
                accuracy_score(y[val_idx], predictions)
            ),
            "precision": float(
                precision_score(y[val_idx], predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(y[val_idx], predictions, zero_division=0)
            ),
            "macro_f1": float(
                f1_score(
                    y[val_idx],
                    predictions,
                    average="macro",
                    zero_division=0
                )
            ),
            "log_loss": float(
                log_loss(y[val_idx], probabilities, labels=[0, 1])
            ),
            "fit_seconds": fit_seconds
        }

        folds.append(metrics)

        print(f"Fold {fold}: Macro F1 = {metrics['macro_f1']:.4f}")

    return folds

def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PathSnap Logistic Regression"
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Folder containing train and student_test"
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Folder to save model and metrics"
    )

    args = parser.parse_args()

    # Create output folder
    args.output.mkdir(parents=True, exist_ok=True)

    # Load training and student-testing datasets
    print("Loading training trips...")
    x, y, groups, train_info = collect(args.data / "train")

    print("Loading student-testing trips...")
    x_test, y_test, test_groups, test_info = collect(
        args.data / "student_test"
    )

    print(f"Training rows: {len(y)}")
    print(f"Training trips: {len(np.unique(groups))}")
    print(f"Student-test rows: {len(y_test)}")
    print(f"Student-test trips: {len(np.unique(test_groups))}")

    # Run cross-validation
    print("\nRunning GroupKFold cross-validation...")

    folds = cross_validate_model(x, y, groups)

    mean_cv_f1 = float(
        np.mean([fold["macro_f1"] for fold in folds])
    )

    print(f"\nMean CV Macro F1: {mean_cv_f1:.4f}")

    # Train final model using all training data
    print("\nTraining final Logistic Regression model...")

    final_model = build_model()

    start = time.perf_counter()
    final_model.fit(x, y)
    training_seconds = time.perf_counter() - start

    # Evaluate on student-testing data
    print("\nEvaluating on student-test trips...")

    start = time.perf_counter()

    probabilities = final_model.predict_proba(x_test)[:, 1]

    inference_seconds = time.perf_counter() - start

    predictions = (probabilities >= 0.5).astype(int)

    student_metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "macro_f1": float(
            f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )
        ),
        "log_loss": float(
            log_loss(y_test, probabilities, labels=[0, 1])
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        ).tolist(),
        "candidate_rows": int(len(y_test)),
        "positive_candidate_rate": float(np.mean(y_test)),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds
    }

    # Save trained model
    model_path = args.output / "logistic_regression.joblib"

    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "parameters": {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs"
            },
            "label": "reference_candidate"
        },
        model_path,
        compress=3
    )

    # Prepare evaluation report
    report = {
        "model": "LogisticRegression",
        "seed": 42,
        "features": FEATURES,
        "training_rows": int(len(y)),
        "training_trips": int(len(np.unique(groups))),
        "student_test_rows": int(len(y_test)),
        "student_test_trips": int(len(np.unique(test_groups))),
        "cv": {
            "split": "GroupKFold by trip",
            "folds": folds,
            "mean_macro_f1": mean_cv_f1
        },
        "student_test": student_metrics,
        "data_quality": {
            "train_trip_inventory": train_info,
            "student_test_trip_inventory": test_info,
            "train_mean_candidate_miss_rate": float(
                np.mean([item["miss_rate"] for item in train_info])
            ),
            "student_test_mean_candidate_miss_rate": float(
                np.mean([item["miss_rate"] for item in test_info])
            )
        }
    }

    # Save metrics as JSON
    metrics_path = args.output / "metrics.json"

    metrics_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    print("\nStudent-test results:")
    print(json.dumps(student_metrics, indent=2))

    print(f"\nModel saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    main()