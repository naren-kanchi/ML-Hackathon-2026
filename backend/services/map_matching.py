"""Candidate scoring and route reconstruction for one PathSnap trip.

Prediction reads only .track, .nodes and .arcs. Reference .route files are
opened separately after inference, only to produce aggregate metrics.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.spatial import cKDTree

LAT_METRES = 111_195.0
FEATURE_NAMES = [
    "distance_m", "heading_error_rad", "gps_speed_mps", "gps_bearing_sin",
    "gps_bearing_cos", "segment_length_m", "segment_bearing_sin",
    "segment_bearing_cos", "candidate_degree_from", "candidate_degree_to",
    "endpoint_min_distance_m", "endpoint_max_distance_m", "time_delta_s",
]


def _read_columns(path: Path, dtype: Any) -> np.ndarray:
    values = np.loadtxt(path, dtype=dtype, ndmin=2)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Empty or malformed data file: {path.name}")
    return values


def _local_xy(lonlat: np.ndarray, latitude_origin: float) -> np.ndarray:
    scale_x = LAT_METRES * math.cos(math.radians(latitude_origin))
    return np.column_stack((lonlat[:, 0] * scale_x, lonlat[:, 1] * LAT_METRES))


def extract_trip_features(trip_dir: Path, radius_m: float = 80.0,
                          max_candidates: int = 40) -> tuple[np.ndarray, list[np.ndarray], dict[str, Any]]:
    """Build candidate features for every GPS point without reading labels."""
    trip_id = trip_dir.name
    track = _read_columns(trip_dir / f"{trip_id}.track", float)
    nodes = _read_columns(trip_dir / f"{trip_id}.nodes", float)
    arcs = _read_columns(trip_dir / f"{trip_id}.arcs", int)
    if track.shape[1] < 3 or nodes.shape[1] < 2 or arcs.shape[1] < 2:
        raise ValueError(f"Trip {trip_id} has an unexpected column count")
    if len(track) < 2:
        raise ValueError(f"Trip {trip_id} needs at least two GPS observations")

    lat0 = float(np.nanmedian(track[:, 1]))
    gps_xy = _local_xy(track[:, :2], lat0)
    node_xy = _local_xy(nodes[:, :2], lat0)
    valid = ((arcs[:, 0] >= 0) & (arcs[:, 1] >= 0) &
             (arcs[:, 0] < len(nodes)) & (arcs[:, 1] < len(nodes)))
    arc_ids = np.flatnonzero(valid)
    ends = arcs[valid, :2]
    if not len(ends):
        raise ValueError(f"Trip {trip_id} has no valid road arcs")
    p0, p1 = node_xy[ends[:, 0]], node_xy[ends[:, 1]]
    vectors = p1 - p0
    segment_lengths = np.maximum(np.linalg.norm(vectors, axis=1), 0.1)
    segment_bearings = np.arctan2(vectors[:, 1], vectors[:, 0])
    midpoints = (p0 + p1) / 2.0
    degrees = np.bincount(ends.ravel(), minlength=len(nodes))

    timestamps = track[:, 2]
    time_delta = np.diff(timestamps, prepend=timestamps[0])
    time_delta[~np.isfinite(time_delta) | (time_delta <= 0)] = 1.0
    motion = np.diff(gps_xy, axis=0, prepend=gps_xy[:1])
    gps_bearing = np.arctan2(motion[:, 1], motion[:, 0])
    gps_speed = np.linalg.norm(motion, axis=1) / time_delta
    road_tree = cKDTree(midpoints)
    features_by_observation: list[np.ndarray] = []
    arcs_by_observation: list[np.ndarray] = []

    for obs_index, point in enumerate(gps_xy):
        candidates = np.asarray(road_tree.query_ball_point(point, radius_m), dtype=int)
        if not len(candidates):
            candidates = np.asarray([int(road_tree.query(point)[1])])
        distances = np.linalg.norm(midpoints[candidates] - point, axis=1)
        if len(candidates) > max_candidates:
            keep = np.argsort(distances)[:max_candidates]
            candidates, distances = candidates[keep], distances[keep]

        delta = np.arctan2(np.sin(segment_bearings[candidates] - gps_bearing[obs_index]),
                           np.cos(segment_bearings[candidates] - gps_bearing[obs_index]))
        endpoint0 = np.linalg.norm(p0[candidates] - point, axis=1)
        endpoint1 = np.linalg.norm(p1[candidates] - point, axis=1)
        features = np.column_stack((
            distances, np.abs(delta), np.full(len(candidates), gps_speed[obs_index]),
            np.full(len(candidates), np.sin(gps_bearing[obs_index])),
            np.full(len(candidates), np.cos(gps_bearing[obs_index])),
            segment_lengths[candidates], np.sin(segment_bearings[candidates]),
            np.cos(segment_bearings[candidates]), degrees[ends[candidates, 0]],
            degrees[ends[candidates, 1]], np.minimum(endpoint0, endpoint1),
            np.maximum(endpoint0, endpoint1), np.full(len(candidates), time_delta[obs_index]),
        )).astype(np.float32)
        features_by_observation.append(features)
        arcs_by_observation.append(arc_ids[candidates])

    metadata = {
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "midpoints": midpoints,
        "arc_ids": arc_ids,
        "arc_ends": ends,
        "features_by_observation": features_by_observation,
        "candidate_arcs_by_observation": arcs_by_observation,
    }
    return track, features_by_observation, metadata


def _viterbi(candidate_arcs: list[np.ndarray], emissions: list[np.ndarray],
             arc_ends: np.ndarray, track: np.ndarray) -> tuple[list[int], float]:
    """Decode candidate states with emissions plus movement/connectivity scores."""
    states = [np.asarray(row, dtype=int) for row in candidate_arcs]
    log_emissions = [np.log(np.clip(np.asarray(row), 1e-8, 1.0)) for row in emissions]
    scores = log_emissions[0].copy()
    backpointers: list[np.ndarray] = []
    # Candidate sets are bounded to 40 states per observation.
    for t in range(1, len(states)):
        prev_states, curr_states = states[t - 1], states[t]
        prev_arcs, curr_arcs = arc_ends[prev_states], arc_ends[curr_states]
        # Movement consistency uses direct arc connectivity and adjusts its
        # strength for long observation gaps.
        connected = prev_arcs[:, 1, None] == curr_arcs[None, :, 0]
        same_arc = prev_states[:, None] == curr_states[None, :]
        elapsed = max(float(track[t, 2] - track[t - 1, 2]), 1.0)
        penalty = 1.25 if elapsed < 30.0 else 0.25
        transition = np.where(connected | same_arc, 0.8, -penalty)
        candidate_scores = scores[:, None] + transition
        best_previous = np.argmax(candidate_scores, axis=0)
        scores = candidate_scores[best_previous, np.arange(len(curr_states))] + log_emissions[t]
        backpointers.append(best_previous)
    last = int(np.argmax(scores))
    decoded = [last]
    for pointers in reversed(backpointers):
        last = int(pointers[last])
        decoded.append(last)
    decoded.reverse()
    path = [int(states[t][decoded[t]]) for t in range(len(states))]
    confidence = float(np.mean([emissions[t][decoded[t]] for t in range(len(states))]))
    return path, confidence


def predict_trip(trip_dir: Path, model_bundle: dict[str, Any]) -> dict[str, Any]:
    track, feature_rows, metadata = extract_trip_features(trip_dir)
    model = model_bundle["model"]
    if list(model_bundle.get("features", FEATURE_NAMES)) != FEATURE_NAMES:
        raise ValueError("Saved model feature order does not match API feature extraction")
    emissions: list[np.ndarray] = []
    for row in feature_rows:
        probabilities = model.predict_proba(row)
        classes = list(model.classes_)
        positive_column = classes.index(1) if 1 in classes else None
        if positive_column is None:
            scores = np.zeros(len(row), dtype=float)
        else:
            scores = probabilities[:, positive_column]
        # The classifier estimates candidate match probability; normalize
        # within each observation before sequence decoding.
        total = float(scores.sum())
        emissions.append(scores / total if total > 0 else np.full(len(scores), 1.0 / len(scores)))

    # Convert graph node endpoints to arc-indexed endpoints for the decoder.
    arc_ends_by_id = np.zeros((len(metadata["arcs"]), 2), dtype=int)
    arc_ends_by_id[metadata["arc_ids"]] = metadata["arc_ends"]
    decoded, mean_confidence = _viterbi(metadata["candidate_arcs_by_observation"], emissions,
                                        arc_ends_by_id, track)
    raw = [int(metadata["candidate_arcs_by_observation"][i][int(np.argmax(emissions[i]))])
           for i in range(len(emissions))]
    arc_ids = metadata["arc_ids"]
    arc_rows = metadata["arcs"]
    endpoints = metadata["nodes"]

    def route_geometry(ids: list[int]) -> list[dict[str, Any]]:
        compact = [arc for i, arc in enumerate(ids) if i == 0 or arc != ids[i - 1]]
        return [{"arc_id": arc, "coordinates": endpoints[arc_rows[arc, :2], :2].tolist()}
                for arc in compact]

    observation_predictions = []
    for i, arc_id in enumerate(decoded):
        decoded_candidate = int(np.flatnonzero(
            metadata["candidate_arcs_by_observation"][i] == arc_id
        )[0])
        observation_predictions.append({
            "timestamp": float(track[i, 2]),
            "gps": [float(track[i, 0]), float(track[i, 1])],
            "arc_id": int(arc_id),
            "confidence": float(emissions[i][decoded_candidate]),
            "classifier_arc_id": int(raw[i]),
        })
    # Route continuity uses directed endpoint agreement for successive output arcs.
    continuity = float(np.mean([
        arc_rows[a, 1] == arc_rows[b, 0] for a, b in zip(decoded, decoded[1:])
    ])) if len(decoded) > 1 else 1.0
    return {
        "trip_id": trip_dir.name,
        "observation_count": int(len(track)),
        "predicted_arc_ids": decoded,
        "classifier_arc_ids": raw,
        "observations": observation_predictions,
        "predicted_route": route_geometry(decoded),
        "classifier_route": route_geometry(raw),
        "mean_emission_confidence": mean_confidence,
        "route_continuity": continuity,
        "candidate_miss_rate": None,
        "evaluation": None,
        "_metadata": metadata,
    }


def evaluate_reference(prediction: dict[str, Any], trip_dir: Path) -> dict[str, Any] | None:
    """Score projected reference labels; never return labels or route IDs."""
    route_path = trip_dir / f"{trip_dir.name}.route"
    if not route_path.is_file():
        return None
    metadata = prediction["_metadata"]
    reference = np.loadtxt(route_path, dtype=int, ndmin=1).ravel()
    route_arcs = metadata["arcs"]
    arc_ids = metadata["arc_ids"]
    valid_ref = reference[(reference >= 0) & (reference < len(route_arcs))]
    valid_ref = valid_ref[np.isin(valid_ref, arc_ids)]
    if not len(valid_ref):
        return {"evaluated_observations": 0, "candidate_miss_rate": 1.0}
    gps = _local_xy(metadata["track"][:, :2], float(np.nanmedian(metadata["track"][:, 1])))
    midpoints = metadata["midpoints"]
    positions = {int(arc): int(np.where(arc_ids == arc)[0][0]) for arc in np.unique(valid_ref)}
    labels_at: dict[int, tuple[float, int]] = {}
    for arc in valid_ref:
        midpoint_index = positions[int(arc)]
        distances = np.linalg.norm(gps - midpoints[midpoint_index], axis=1)
        obs = int(np.argmin(distances))
        if obs not in labels_at or distances[obs] < labels_at[obs][0]:
            labels_at[obs] = (float(distances[obs]), int(arc))
    truth, raw_pred, decoded_pred, covered = [], [], [], []
    for obs, (_, label) in labels_at.items():
        candidates = metadata["candidate_arcs_by_observation"][obs]
        truth.append(label)
        raw_pred.append(prediction["classifier_arc_ids"][obs])
        decoded_pred.append(prediction["predicted_arc_ids"][obs])
        covered.append(label in candidates)
    truth_a, raw_a, decoded_a, covered_a = map(np.asarray, (truth, raw_pred, decoded_pred, covered))
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    continuity_flags = [route_arcs[a, 1] == route_arcs[b, 0]
                        for a, b in zip(decoded_a, decoded_a[1:])]
    return {
        "evaluated_observations": int(len(truth_a)),
        "candidate_miss_rate": float(1.0 - np.mean(covered_a)),
        "classifier_top1_accuracy": float(accuracy_score(truth_a, raw_a)),
        "classifier_macro_f1": float(f1_score(truth_a, raw_a, average="macro", zero_division=0)),
        "viterbi_top1_accuracy": float(accuracy_score(truth_a, decoded_a)),
        "viterbi_macro_f1": float(f1_score(truth_a, decoded_a, average="macro", zero_division=0)),
        "viterbi_precision_macro": float(precision_score(truth_a, decoded_a, average="macro", zero_division=0)),
        "viterbi_recall_macro": float(recall_score(truth_a, decoded_a, average="macro", zero_division=0)),
        "viterbi_route_continuity": float(np.mean(continuity_flags)) if continuity_flags else 1.0,
    }
