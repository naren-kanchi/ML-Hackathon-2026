"""
PathSnap - Geographical checks (Section 5) and reference-route limitations (Section 6).

Usage:
    python geo_route_analysis.py <dataset_root> <output_dir> [trip_id ...]

Optional trip_ids get map outputs (default: 00000002 00000033 00000054 00000056 00000079 00000090).

What it does
  Section 5
    * Projects every trip into metres. The script uses the trip's own UTM zone when
      pyproj is installed. Otherwise it uses a local tangent-plane (equirectangular)
      projection centred on the trip.
    * Compares projected distances with haversine distances so the choice is backed
      by numbers, and shows how wrong raw degree-Euclidean distance is.
    * Checks coordinate order: GPS points must lie close to the road network.
  Section 6
    * Distance from every GPS point to the nearest arc of its reference route.
      Exact point-to-segment distance in metres.
    * Distance to the nearest road of any kind. When a point is clearly closer to
      another road than to its route, that road is a source of ambiguity.
    * Route coverage: the share of route arcs with a GPS point nearby, and the share
      of GPS points far from the route. Both indicate a route that only partly
      covers the drive.
    * Emission sigma estimate (Newson & Krumm: 1.4826 * median distance).
      Computed from TRAIN trips only (leakage rule, Section 7).
  Outputs
    trip_geo.csv, geo_summary.md, plots/*.png, maps/*.html (Folium, if installed),
    maps/*.png (static, always)
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pyproj
except ImportError:
    pyproj = None
try:
    import folium
except ImportError:
    folium = None

R_EARTH = 6371008.8
DEFAULT_MAP_TRIPS = ["00000002", "00000033", "00000054", "00000056", "00000079", "00000090"]
FAR_M = 100          # a GPS point farther than this from its route counts as "far"
COVER_M = 50         # a route arc counts as "covered" if a GPS point lies within this
AMBIG_M = 10         # point is "ambiguous" if another road is closer than route by > this
DENSIFY_M = 5        # spacing for the nearest-road KD-tree (max error DENSIFY_M/2)
STAT_WIN_S = 60      # a point is "stationary" if every GPS point within +/- STAT_WIN_S/2 s
STAT_SPAN_M = 30     #   fits inside a STAT_SPAN_M x STAT_SPAN_M box (parked, GPS drifting)

C_GPS, C_ROUTE, C_NET, C_INK, C_MUTED = "#2a78d6", "#eb6834", "#b8b7b0", "#0b0b0b", "#52514e"


# ----------------------------------------------------------------------------- io
def load_trip(d, trip):
    f = lambda e: os.path.join(d, f"{trip}.{e}")
    track = np.loadtxt(f("track"), ndmin=2)              # lon, lat, t
    nodes = np.loadtxt(f("nodes"), ndmin=2)              # lon, lat
    arcs = np.loadtxt(f("arcs"), dtype=np.int64, ndmin=2)
    route = (np.loadtxt(f("route"), dtype=np.int64, ndmin=1)
             if os.path.getsize(f("route")) else np.array([], np.int64))
    return track, nodes, arcs, route


def stationary_mask(G, t):
    """True where the vehicle is parked: within a +/-30 s window the points stay
    inside a 30 m box. Time-based, so it also works across sampling gaps."""
    df = pd.DataFrame({"x": G[:, 0], "y": G[:, 1]}, index=pd.to_datetime(t, unit="s"))
    w = f"{STAT_WIN_S}s"
    span = lambda c: (df[c].rolling(w, center=True).max() - df[c].rolling(w, center=True).min()).values
    return (span("x") < STAT_SPAN_M) & (span("y") < STAT_SPAN_M)


# ----------------------------------------------------------------- distances
def haversine_m(lon1, lat1, lon2, lat2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R_EARTH * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class Projector:
    """lon/lat -> metres for one trip."""

    def __init__(self, lon0, lat0):
        self.lon0, self.lat0 = lon0, lat0
        if pyproj is not None:
            zone = int((lon0 + 180) // 6) + 1
            epsg = (32600 if lat0 >= 0 else 32700) + zone
            self.t = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
            self.name = f"UTM zone {zone}{'N' if lat0 >= 0 else 'S'} (EPSG:{epsg})"
        else:
            self.t = None
            self.name = "local tangent plane (equirectangular, centred on trip)"

    def __call__(self, lon, lat):
        if self.t is not None:
            return np.column_stack(self.t.transform(lon, lat))
        x = R_EARTH * np.radians(lon - self.lon0) * np.cos(np.radians(self.lat0))
        y = R_EARTH * np.radians(lat - self.lat0)
        return np.column_stack([x, y])


def point_to_segments(P, A, B, chunk=2048):
    """Min distance from each point in P to any segment A[i]-B[i] (all in metres).
    Returns (distance, index of nearest segment)."""
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-12)
    dmin = np.empty(len(P)); imin = np.empty(len(P), np.int64)
    for s in range(0, len(P), chunk):
        p = P[s:s + chunk, None, :]
        t = np.clip(((p - A) * AB).sum(2) / L2, 0, 1)
        d = np.sqrt(((p - (A + t[..., None] * AB)) ** 2).sum(2))
        imin[s:s + chunk] = d.argmin(1)
        dmin[s:s + chunk] = d[np.arange(len(d)), imin[s:s + chunk]]
    return dmin, imin


def densify(A, B, step):
    L = np.sqrt(((B - A) ** 2).sum(1))
    n = np.maximum(np.ceil(L / step).astype(int), 1)
    idx = np.repeat(np.arange(len(A)), n + 1)
    frac = np.concatenate([np.linspace(0, 1, k + 1) for k in n])
    return A[idx] + frac[:, None] * (B - A)[idx]


# ------------------------------------------------------------------- per trip
def analyse(part, trip, d, rng):
    track, nodes, arcs, route = load_trip(d, trip)
    lon, lat = track[:, 0], track[:, 1]
    prj = Projector(np.median(lon), np.median(lat))
    G = prj(lon, lat)
    N = prj(nodes[:, 0], nodes[:, 1])
    r = {"part": part, "trip": trip, "projection": prj.name, "n_gps": len(track)}
    stat = stationary_mask(G, track[:, 2])
    r["pct_stationary"] = stat.mean() * 100
    step = np.sqrt((np.diff(G, axis=0) ** 2).sum(1))
    r["path_km_all"] = step.sum() / 1000
    r["path_km_moving"] = step[~(stat[1:] & stat[:-1])].sum() / 1000

    # --- Section 5: projection accuracy vs haversine, and the naive-degree error
    i, j = rng.integers(0, len(G), (2, min(3000, len(G) * 2)))
    keep = i != j
    i, j = i[keep], j[keep]
    hv = haversine_m(lon[i], lat[i], lon[j], lat[j])
    ok = hv > 50
    proj_d = np.sqrt(((G[i] - G[j]) ** 2).sum(1))
    naive_d = np.sqrt((lon[i] - lon[j]) ** 2 + (lat[i] - lat[j]) ** 2) * 111_195  # deg*m
    r["proj_err_max_pct"] = np.abs(proj_d[ok] / hv[ok] - 1).max() * 100 if ok.any() else 0
    r["naive_deg_err_median_pct"] = np.median(np.abs(naive_d[ok] / hv[ok] - 1)) * 100 if ok.any() else 0
    r["naive_deg_err_max_pct"] = np.abs(naive_d[ok] / hv[ok] - 1).max() * 100 if ok.any() else 0
    r["abs_lat"] = abs(np.median(lat))

    # --- nearest road of any kind (KD-tree over densified arcs near the track)
    lo, hi = G.min(0) - 500, G.max(0) + 500
    near = (((N[arcs[:, 0]] >= lo) & (N[arcs[:, 0]] <= hi)).all(1)
            | ((N[arcs[:, 1]] >= lo) & (N[arcs[:, 1]] <= hi)).all(1))
    a = arcs[near]
    a = np.unique(np.sort(a, 1), axis=0)              # undirected, no duplicates
    tree = cKDTree(densify(N[a[:, 0]], N[a[:, 1]], DENSIFY_M))
    d_net, _ = tree.query(G)
    # coordinate-order check: swapped lon/lat would put points ~far from every road
    r["net_dist_median_m"] = np.median(d_net)
    r["net_dist_p95_m"] = np.percentile(d_net, 95)
    r["coord_order_ok"] = bool(np.median(d_net) < 100)

    # --- Section 6: distance to the reference route
    out = {"stat": stat, "G": G, "N": N, "arcs": arcs, "route": route, "prj": prj, "track": track, "nodes": nodes}
    if len(route):
        ra = arcs[route]
        d_rt, seg = point_to_segments(G, N[ra[:, 0]], N[ra[:, 1]])
        out["d_rt"] = d_rt
        # route coverage: a route arc is covered when a GPS point is within COVER_M
        # (measured from each arc's midpoint to the nearest GPS point, so short arcs
        #  that are never any single point's "nearest" are still counted correctly)
        mid = (N[ra[:, 0]] + N[ra[:, 1]]) / 2
        cov = cKDTree(G).query(mid)[0] <= COVER_M
        far = d_rt > FAR_M
        r.update({
            "rt_dist_median_m": np.median(d_rt),
            "rt_dist_median_moving_m": np.median(d_rt[~stat]) if (~stat).any() else np.nan,
            f"pct_far_pts_stationary": (stat[far].mean() * 100) if far.any() else np.nan,
            "rt_dist_p90_m": np.percentile(d_rt, 90),
            "rt_dist_p95_m": np.percentile(d_rt, 95),
            "rt_dist_max_m": d_rt.max(),
            f"pct_gps_gt{FAR_M}m_from_route": far.mean() * 100,
            "pct_gps_gt20m_from_route": (d_rt > 20).mean() * 100,
            f"pct_route_arcs_covered_{COVER_M}m": cov.mean() * 100,
            f"pct_gps_ambiguous_{AMBIG_M}m": ((d_rt - d_net) > AMBIG_M).mean() * 100,
            "first_gps_on_route_idx": int(np.argmax(~far)) if (~far).any() else -1,
            "last_gps_on_route_idx": int(len(far) - 1 - np.argmax(~far[::-1])) if (~far).any() else -1,
        })
        # longest contiguous run of 'far' points: tells partial coverage apart from jitter
        runs, cur = 0, 0
        for x in far:
            cur = cur + 1 if x else 0
            runs = max(runs, cur)
        r["longest_far_run_pts"] = runs
    return r, out


# ----------------------------------------------------------------------- maps
def static_map(trip, o, path):
    G, N, arcs, route = o["G"], o["N"], o["arcs"], o["route"]
    lo, hi = G.min(0) - 300, G.max(0) + 300
    m = (((N[arcs[:, 0]] >= lo) & (N[arcs[:, 0]] <= hi)).all(1))
    fig, ax = plt.subplots(figsize=(8, 8))
    segs = np.stack([N[arcs[m, 0]], N[arcs[m, 1]]], 1)
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(segs, colors=C_NET, linewidths=0.4, zorder=1))
    if len(route):
        ra = arcs[route]
        ax.add_collection(LineCollection(np.stack([N[ra[:, 0]], N[ra[:, 1]]], 1),
                                         colors=C_ROUTE, linewidths=2, zorder=2, label="Reference route"))
    ax.plot(G[:, 0], G[:, 1], ".", ms=2, color=C_GPS, zorder=3, label="GPS points")
    ax.plot(*G[0], "o", ms=8, color=C_INK, zorder=4); ax.annotate("start", G[0], xytext=(5, 5), textcoords="offset points", color=C_INK)
    ax.plot(*G[-1], "s", ms=8, color=C_INK, zorder=4); ax.annotate("end", G[-1], xytext=(5, 5), textcoords="offset points", color=C_INK)
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
    ax.set_title(f"Trip {trip} - {o['prj'].name}", color=C_INK, fontsize=11, loc="left")
    ax.set_xlabel("x (m)", color=C_MUTED); ax.set_ylabel("y (m)", color=C_MUTED)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    for s in ax.spines.values(): s.set_color(C_NET)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def folium_map(trip, o, path):
    """Interactive map. NOTE: Folium wants [lat, lon]; the files store lon, lat."""
    tr, nd, arcs, route = o["track"], o["nodes"], o["arcs"], o["route"]
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, tiles="OpenStreetMap")
    # nearby network (limit size so the HTML stays light)
    G, N = o["G"], o["N"]
    lo, hi = G.min(0) - 300, G.max(0) + 300
    near = np.where((((N[arcs[:, 0]] >= lo) & (N[arcs[:, 0]] <= hi)).all(1)))[0][:30000]
    net = folium.FeatureGroup("Road network", show=True)
    folium.PolyLine([[[nd[a, 1], nd[a, 0]], [nd[b, 1], nd[b, 0]]] for a, b in arcs[near]],
                    color=C_NET, weight=1).add_to(net)
    net.add_to(m)
    if len(route):
        rg = folium.FeatureGroup("Reference route")
        for k, (a, b) in enumerate(arcs[route]):
            folium.PolyLine([[nd[a, 1], nd[a, 0]], [nd[b, 1], nd[b, 0]]], color=C_ROUTE, weight=4,
                            tooltip=f"route pos {k}, arc {route[k]}").add_to(rg)
        rg.add_to(m)
    gg = folium.FeatureGroup("GPS points")
    d_rt = o.get("d_rt", np.full(len(tr), np.nan))
    for k in range(0, len(tr), max(1, len(tr) // 3000)):
        folium.CircleMarker([tr[k, 1], tr[k, 0]], radius=2, color=C_GPS, fill=True, weight=0, fill_opacity=0.8,
                            tooltip=f"t={tr[k, 2]:.0f}s, {d_rt[k]:.1f} m from route").add_to(gg)
    gg.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="start").add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="end").add_to(m)
    folium.LayerControl().add_to(m)
    m.save(path)


# ---------------------------------------------------------------------- plots
def style(ax, title, xlabel, ylabel):
    ax.set_title(title, color=C_INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=C_MUTED); ax.set_ylabel(ylabel, color=C_MUTED)
    ax.tick_params(colors=C_MUTED, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(C_NET)


def make_plots(df, train_d, out):
    os.makedirs(out, exist_ok=True)
    # 1. GPS -> route distance distribution (train only)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(np.clip(train_d, 0, 100), bins=100, color=C_GPS, rwidth=0.9)
    med = np.median(train_d)
    ax.axvline(med, color=C_INK, lw=1, ls="--")
    ax.annotate(f"median {med:.1f} m", (med, ax.get_ylim()[1] * 0.9), xytext=(6, 0), textcoords="offset points", color=C_INK)
    style(ax, "Distance from each GPS point to its reference route (train, clipped at 100 m)",
          "distance (m)", "GPS points")
    fig.tight_layout(); fig.savefig(os.path.join(out, "gps_to_route_distance_hist.png"), dpi=130); plt.close(fig)

    # 2. naive degree distance error vs latitude
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df.abs_lat, df.naive_deg_err_median_pct, "o", ms=6, color=C_ROUTE, label="Euclidean on degrees")
    ax.plot(df.abs_lat, df.proj_err_max_pct, "o", ms=6, color=C_GPS, label="Projected (worst pair)")
    style(ax, "Distance error vs haversine, by trip latitude", "|latitude| (deg)", "error (%)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "distance_method_error.png"), dpi=130); plt.close(fig)

    # 3. per-trip p95 distance to route
    s = df.dropna(subset=["rt_dist_p95_m"]).sort_values("rt_dist_p95_m")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(range(len(s)), s.rt_dist_p95_m, color=[C_GPS if p == "train" else C_ROUTE for p in s.part], width=0.8)
    ax.set_yscale("log")
    style(ax, "95th-percentile GPS-to-route distance per trip (blue = train, orange = student_test)",
          "trips (sorted)", "p95 distance (m, log)")
    ax.set_xticks([])
    fig.tight_layout(); fig.savefig(os.path.join(out, "route_distance_p95_per_trip.png"), dpi=130); plt.close(fig)


# ---------------------------------------------------------------------- main
def main(root, out, map_trips):
    rng = np.random.default_rng(0)
    os.makedirs(os.path.join(out, "maps"), exist_ok=True)
    rows, train_d = [], []
    for part in sorted(p for p in os.listdir(root) if os.path.isdir(os.path.join(root, p))):
        for trip in sorted(os.listdir(os.path.join(root, part))):
            d = os.path.join(root, part, trip)
            if not os.path.isdir(d):
                continue
            r, o = analyse(part, trip, d, rng)
            rows.append(r)
            if part == "train" and "d_rt" in o:
                train_d.append(o["d_rt"])
            if trip in map_trips:
                static_map(trip, o, os.path.join(out, "maps", f"trip_{trip}.png"))
                if folium is not None:
                    folium_map(trip, o, os.path.join(out, "maps", f"trip_{trip}.html"))
            print(f"{part}/{trip}: median {r.get('rt_dist_median_m', np.nan):.1f} m to route", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "trip_geo.csv"), index=False)
    train_d = np.concatenate(train_d)
    make_plots(df, train_d, os.path.join(out, "plots"))

    sigma = 1.4826 * np.median(train_d)
    tr = df[df.part == "train"]
    partial = df[(df[f"pct_route_arcs_covered_{COVER_M}m"] < 90) | (df[f"pct_gps_gt{FAR_M}m_from_route"] > 5)]
    L = ["# PathSnap - Geographical checks and route limitations\n",
         "## Section 5: Geographical data\n",
         f"**Projection used:** {'UTM, each trip in its own zone' if df.projection.str.startswith('UTM').all() else 'local tangent plane, one per trip'} "
         f"({df.projection.nunique()} different zones across the dataset; see `projection` column in trip_geo.csv).  ",
         f"Worst projected-vs-haversine error over all trips: **{df.proj_err_max_pct.max():.3f}%**.  ",
         f"Euclidean distance on raw degrees (x 111 km): median error **{df.naive_deg_err_median_pct.median():.1f}%**, "
         f"worst **{df.naive_deg_err_max_pct.max():.1f}%** (grows with latitude, see plots/distance_method_error.png).  ",
         f"Coordinate order (lon, lat) confirmed for **{df.coord_order_ok.sum()}/{len(df)}** trips "
         f"(median GPS-to-nearest-road distance {df.net_dist_median_m.median():.1f} m).\n",
         "## Section 6: GPS vs reference route\n",
         "Distance from each GPS point to the nearest arc of its reference route, train trips only:\n",
         pd.Series(train_d).describe(percentiles=[.5, .9, .95, .99]).round(2).rename("metres").to_markdown(), "",
         f"**Emission sigma estimate (Newson & Krumm, 1.4826 x median, train only): {sigma:.2f} m**\n",
         f"**Stationary periods** (points staying inside a {STAT_SPAN_M} m box for {STAT_WIN_S} s): "
         f"{(df.pct_stationary > 0).sum()} trips have some; median {df.pct_stationary.median():.1f}% of points, "
         f"max {df.pct_stationary.max():.1f}%. Among trips with far-from-route points, "
         f"{(df.pct_far_pts_stationary > 50).sum()} have mostly stationary far points (parked off the network).\n",
         df.nlargest(8, "pct_stationary")[["part", "trip", "pct_stationary", "path_km_all", "path_km_moving",
                                           "rt_dist_median_m", "rt_dist_median_moving_m"]].round(1).to_markdown(index=False), "",
         f"Points whose nearest road is not their route road (by >{AMBIG_M} m): "
         f"median {tr[f'pct_gps_ambiguous_{AMBIG_M}m'].median():.1f}% per trip, max {tr[f'pct_gps_ambiguous_{AMBIG_M}m'].max():.1f}%.\n",
         f"### Trips where the route covers the drive only partly ({len(partial)})\n",
         f"Criteria: < 90% of route arcs have a GPS point within {COVER_M} m, or > 5% of GPS points are > {FAR_M} m from the route.\n",
         partial[["part", "trip", "n_gps", "rt_dist_median_m", "rt_dist_p95_m", f"pct_gps_gt{FAR_M}m_from_route",
                  f"pct_route_arcs_covered_{COVER_M}m", "longest_far_run_pts", "pct_far_pts_stationary", "first_gps_on_route_idx",
                  "last_gps_on_route_idx"]].round(1).to_markdown(index=False), ""]
    with open(os.path.join(out, "geo_summary.md"), "w") as fh:
        fh.write("\n".join(L))
    print(f"\nprojection: {df.projection.nunique()} different per-trip projections (e.g. {df.projection.iloc[0]})\nsigma (train) = {sigma:.2f} m\nfolium maps: {'yes' if folium else 'skipped (pip install folium)'}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:] or DEFAULT_MAP_TRIPS)