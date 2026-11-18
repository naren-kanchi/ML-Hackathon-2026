"""
PathSnap - Initial dataset exploration (covers every check in Section 4 of the
Dataset Exploration document).

Usage:
    python dataset_inventory.py <dataset_root> <output_dir>

<dataset_root> must contain partition folders (e.g. train/, student_test/),
each holding one folder per trip with <trip>.track/.nodes/.arcs/.route.

Outputs (in <output_dir>):
    trip_inventory.csv     one row per trip, all metrics
    file_inventory.csv     one row per (trip, file type): exists, size, rows, cols, dtype
    eda_summary.md         checklist-ordered summary + anomaly list
"""
import os
import sys

import numpy as np
import pandas as pd

try:
    import networkx as nx
except ImportError:  # connectivity checks are skipped without networkx
    nx = None

EXT_SCHEMA = {
    # ext: (expected n columns, column names, expected kind)
    "track": (3, ["longitude", "latitude", "timestamp_s"], "float"),
    "nodes": (2, ["longitude", "latitude"], "float"),
    "arcs": (2, ["from_node_id", "to_node_id"], "int"),
    "route": (1, ["arc_id"], "int"),
}

# thresholds used for anomaly flags (documented in the summary)
GAP_S = 10            # a time gap larger than this counts as a "gap"
LARGE_GAP_S = 60      # a gap larger than this flags the trip
MAX_SPEED_KMH = 250   # implied speed above this = implausible jump


def haversine_m(lon1, lat1, lon2, lat2):
    R = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(a))


def read_raw(path, ext):
    """Read a headerless whitespace file and describe its schema."""
    ncol_exp, names, kind = EXT_SCHEMA[ext]
    info = {"exists": os.path.exists(path), "size_bytes": np.nan, "rows": 0,
            "cols": np.nan, "cols_ok": False, "dtype_ok": False,
            "non_numeric": 0, "nan_values": 0}
    if not info["exists"]:
        return None, info
    info["size_bytes"] = os.path.getsize(path)
    if info["size_bytes"] == 0:
        return pd.DataFrame(columns=names), info
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    info["rows"], info["cols"] = df.shape
    info["cols_ok"] = df.shape[1] == ncol_exp
    num = df.apply(pd.to_numeric, errors="coerce")
    info["non_numeric"] = int(num.isna().sum().sum() - df.isna().sum().sum())
    info["nan_values"] = int(num.isna().sum().sum())
    if kind == "int":
        info["dtype_ok"] = bool(np.all(np.mod(num.dropna().values, 1) == 0))
    else:
        info["dtype_ok"] = info["non_numeric"] == 0
    if info["cols_ok"]:
        num.columns = names
    return num, info


def analyse_trip(part, trip, d):
    frows, data = [], {}
    for ext in EXT_SCHEMA:
        df, info = read_raw(os.path.join(d, f"{trip}.{ext}"), ext)
        data[ext] = df
        frows.append({"part": part, "trip": trip, "file": ext, **info})

    r = {"part": part, "trip": trip,
         "all_files_present": all(f["exists"] for f in frows),
         "schema_ok": all(f["cols_ok"] and f["dtype_ok"] for f in frows if f["rows"] > 0)}
    tr, nd, ar, rt = (data[k] for k in ("track", "nodes", "arcs", "route"))

    # ---------------- GPS observations ----------------
    if tr is not None and len(tr):
        lon, lat, t = tr["longitude"].values, tr["latitude"].values, tr["timestamp_s"].values
        dt, step = np.diff(t), haversine_m(lon[:-1], lat[:-1], lon[1:], lat[1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            spd = np.where(dt > 0, step / dt * 3.6, np.nan)
        r.update({
            "n_gps": len(tr),
            "gps_nan_rows": int(tr.isna().any(axis=1).sum()),
            "gps_out_of_range": int(((np.abs(lat) > 90) | (np.abs(lon) > 180)).sum()),
            "gps_null_island": int(((lat == 0) & (lon == 0)).sum()),
            "gps_dup_consecutive_pos": int(((step == 0)).sum()),
            "gps_dup_timestamps": int((dt == 0).sum()),
            "ts_monotonic": bool((dt > 0).all()),
            "ts_backwards": int((dt < 0).sum()),
            "ts_integer": bool(np.all(np.mod(t, 1) == 0)),
            "t_start": t[0], "t_end": t[-1], "duration_s": t[-1] - t[0],
            "dt_median": np.median(dt), "pct_dt_1s": (dt == 1).mean() * 100,
            f"n_gaps_gt{GAP_S}s": int((dt > GAP_S).sum()),
            f"n_gaps_gt{LARGE_GAP_S}s": int((dt > LARGE_GAP_S).sum()),
            "max_gap_s": dt.max(),
            "missing_time_s": float(np.clip(dt - 1, 0, None).sum()),
            "path_km": step.sum() / 1000,
            "max_step_m": step.max(),
            "max_speed_kmh": np.nanmax(spd),
            f"n_speed_gt{MAX_SPEED_KMH}": int((spd > MAX_SPEED_KMH).sum()),
            "lon_min": lon.min(), "lon_max": lon.max(),
            "lat_min": lat.min(), "lat_max": lat.max(),
            "extent_ew_km": haversine_m(lon.min(), lat.mean(), lon.max(), lat.mean()) / 1000,
            "extent_ns_km": haversine_m(lon.mean(), lat.min(), lon.mean(), lat.max()) / 1000,
        })
    else:
        r["n_gps"] = 0

    # ---------------- Road network ----------------
    if nd is not None and ar is not None and len(nd) and len(ar):
        a = ar.values.astype(np.int64)
        n = len(nd)
        bad_ref = int(((a < 0) | (a >= n)).any(axis=1).sum())
        a_ok = a[((a >= 0) & (a < n)).all(axis=1)]
        deg = np.bincount(a_ok.ravel(), minlength=n)
        arcset = set(map(tuple, a_ok))
        r.update({
            "n_nodes": n, "n_arcs": len(a),
            "arcs_bad_node_ref": bad_ref,
            "isolated_nodes": int((deg == 0).sum()),
            "self_loops": int((a_ok[:, 0] == a_ok[:, 1]).sum()),
            "duplicate_arcs": len(a_ok) - len(arcset),
            "dup_node_coords": int(nd.duplicated().sum()),
            "pct_bidirectional": np.mean([(y, x) in arcset for x, y in a_ok]) * 100,
            "mean_out_degree": len(a_ok) / n,
            "net_lon_min": nd["longitude"].min(), "net_lon_max": nd["longitude"].max(),
            "net_lat_min": nd["latitude"].min(), "net_lat_max": nd["latitude"].max(),
        })
        if tr is not None and len(tr):
            inside = ((tr.longitude.between(r["net_lon_min"], r["net_lon_max"]))
                      & (tr.latitude.between(r["net_lat_min"], r["net_lat_max"])))
            r["pct_gps_inside_network_bbox"] = inside.mean() * 100
        if nx is not None:
            G = nx.DiGraph()
            G.add_nodes_from(range(n))
            G.add_edges_from(map(tuple, a_ok))
            wcc = list(nx.weakly_connected_components(G))
            scc = list(nx.strongly_connected_components(G))
            big_scc = max(scc, key=len)
            r.update({
                "n_weak_components": len(wcc),
                "largest_wcc_pct": max(map(len, wcc)) / n * 100,
                "n_strong_components": len(scc),
                "largest_scc_pct": len(big_scc) / n * 100,
                "n_singleton_scc": sum(len(c) == 1 for c in scc),
            })

    # ---------------- Reference route ----------------
    r["route_available"] = rt is not None and len(rt) > 0
    if r["route_available"] and ar is not None:
        ids = rt["arc_id"].values.astype(np.int64)
        valid = (ids >= 0) & (ids < len(ar))
        ra = ar.values.astype(np.int64)[ids[valid]]
        breaks = (ra[:-1, 1] != ra[1:, 0]) if len(ra) > 1 else np.array([], bool)
        r.update({
            "n_route_arcs": len(ids),
            "route_invalid_ids": int((~valid).sum()),
            "route_unique_arcs": len(np.unique(ids)),
            "route_repeated_arcs": len(ids) - len(np.unique(ids)),
            "route_consecutive_repeat": int((np.diff(ids) == 0).sum()),
            "route_breaks": int(breaks.sum()),
            "route_contiguous_pct": (1 - breaks.mean()) * 100 if len(breaks) else 100.0,
        })
        if len(ra) and tr is not None and len(tr):
            ndv = nd.values
            s, e = ndv[ra[0, 0]], ndv[ra[-1, 1]]
            r["route_start_to_gps_start_m"] = haversine_m(s[0], s[1], tr.longitude.iloc[0], tr.latitude.iloc[0])
            r["route_end_to_gps_end_m"] = haversine_m(e[0], e[1], tr.longitude.iloc[-1], tr.latitude.iloc[-1])
            seg = haversine_m(ndv[ra[:, 0], 0], ndv[ra[:, 0], 1], ndv[ra[:, 1], 0], ndv[ra[:, 1], 1])
            r["route_length_km"] = seg.sum() / 1000
            r["route_vs_gps_length_ratio"] = r["route_length_km"] / max(r.get("path_km", np.nan), 1e-9)
    return r, frows


def flag_anomalies(df):
    flags = []
    for _, r in df.iterrows():
        f = []
        if not r.all_files_present: f.append("missing file")
        if not r.schema_ok: f.append("schema mismatch")
        if r.get("gps_nan_rows", 0) or r.get("gps_out_of_range", 0) or r.get("gps_null_island", 0):
            f.append("invalid GPS values")
        if not r.get("ts_monotonic", True): f.append("non-monotonic timestamps")
        if r.get(f"n_gaps_gt{LARGE_GAP_S}s", 0) > 0: f.append(f"gap >{LARGE_GAP_S}s (max {r.max_gap_s:.0f}s)")
        if r.get(f"n_speed_gt{MAX_SPEED_KMH}", 0) > 0: f.append(f"implausible jumps (max {r.max_speed_kmh:.0f} km/h)")
        if r.get("t_start", 1) > 1: f.append(f"track starts at t={r.t_start:.0f}")
        if not r.route_available: f.append("no route")
        if r.get("route_breaks", 0) > 0: f.append(f"route breaks: {int(r.route_breaks)}")
        if r.get("route_repeated_arcs", 0) > 0: f.append(f"route revisits arcs: {int(r.route_repeated_arcs)}")
        if r.get("pct_gps_inside_network_bbox", 100) < 100: f.append("GPS outside network bbox")
        if r.get("largest_scc_pct", 100) < 90: f.append(f"fragmented network (largest SCC {r.largest_scc_pct:.0f}%)")
        flags.append("; ".join(f))
    df["anomalies"] = flags
    return df


def summary_md(df, files):
    def desc(cols):
        return df[cols].describe().T[["mean", "min", "50%", "max"]].round(2).to_markdown()
    L = ["# PathSnap - Dataset exploration summary\n"]
    L += ["## 1. Trips per partition\n", df.groupby("part").size().rename("trips").to_markdown(), ""]
    L += ["## 2. GPS observations per trip\n", desc(["n_gps"]), ""]
    fs = files.groupby("file").agg(present=("exists", "sum"), total=("exists", "size"),
                                   size_mb_mean=("size_bytes", lambda s: s.mean() / 1e6),
                                   size_mb_max=("size_bytes", lambda s: s.max() / 1e6)).round(3)
    L += ["## 3. File availability and sizes\n", fs.to_markdown(), ""]
    schema = pd.DataFrame([{"file": k, "columns": ", ".join(v[1]), "expected_dtype": v[2]} for k, v in EXT_SCHEMA.items()])
    chk = files[files.rows > 0].groupby("file").agg(cols_ok=("cols_ok", "all"), dtype_ok=("dtype_ok", "all"),
                                                     non_numeric=("non_numeric", "sum")).reset_index()
    L += ["## 4. Data types and column names\n", schema.merge(chk, on="file").to_markdown(index=False), ""]
    L += ["## 5. Missing / invalid GPS observations\n",
          df[["gps_nan_rows", "gps_out_of_range", "gps_null_island", "gps_dup_consecutive_pos", "gps_dup_timestamps"]]
          .sum().rename("total across trips").to_markdown(), ""]
    L += ["## 6. Timestamp ordering and irregular gaps\n",
          f"Trips with non-monotonic timestamps: {(~df.ts_monotonic).sum()}  \n"
          f"Trips with gaps >{GAP_S}s: {(df[f'n_gaps_gt{GAP_S}s'] > 0).sum()}  \n"
          f"Trips with gaps >{LARGE_GAP_S}s: {(df[f'n_gaps_gt{LARGE_GAP_S}s'] > 0).sum()}\n",
          desc(["dt_median", "pct_dt_1s", "max_gap_s", "duration_s"]), ""]
    L += ["## 7. Geographical extent\n", desc(["extent_ew_km", "extent_ns_km", "path_km"]), ""]
    L += ["## 8. Road-network size\n", desc(["n_nodes", "n_arcs", "mean_out_degree"]), ""]
    if "n_strong_components" in df:
        L += ["## 9-10. Connectivity, isolated nodes, disconnected components\n",
              desc(["isolated_nodes", "n_weak_components", "largest_wcc_pct", "n_strong_components",
                    "largest_scc_pct", "n_singleton_scc", "pct_bidirectional", "self_loops", "duplicate_arcs"]), ""]
    an = df[df.anomalies != ""][["part", "trip", "anomalies"]]
    L += [f"## 11. Trips with large gaps or other anomalies ({len(an)} trips)\n",
          f"Thresholds: gap >{LARGE_GAP_S}s, implied speed >{MAX_SPEED_KMH} km/h.\n",
          an.to_markdown(index=False), ""]
    L += ["## 12. Reference route availability and structure\n",
          f"Trips with a route: {df.route_available.sum()} / {len(df)}\n",
          desc(["n_route_arcs", "route_repeated_arcs", "route_breaks", "route_contiguous_pct",
                "route_start_to_gps_start_m", "route_end_to_gps_end_m", "route_vs_gps_length_ratio"]), ""]
    return "\n".join(L)


def main(root, out):
    os.makedirs(out, exist_ok=True)
    rows, frows = [], []
    for part in sorted(p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))):
        for trip in sorted(os.listdir(os.path.join(root, part))):
            d = os.path.join(root, part, trip)
            if os.path.isdir(d):
                r, f = analyse_trip(part, trip, d)
                rows.append(r); frows.extend(f)
                print(f"{part}/{trip}: {r.get('n_gps', 0)} GPS pts, {r.get('n_arcs', 0)} arcs", flush=True)
    df = flag_anomalies(pd.DataFrame(rows))
    files = pd.DataFrame(frows)
    df.to_csv(os.path.join(out, "trip_inventory.csv"), index=False)
    files.to_csv(os.path.join(out, "file_inventory.csv"), index=False)
    with open(os.path.join(out, "eda_summary.md"), "w") as fh:
        fh.write(summary_md(df, files))
    print(f"\nWrote trip_inventory.csv, file_inventory.csv, eda_summary.md to {out}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
