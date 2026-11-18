"""Folium maps of the selected model (tuned LightGBM ranker + tuned Viterbi) on
student-test trips: GPS points, nearby road network, reference route, predicted
route (Viterbi) and classifier-only route, with a legend and layer control.

The prediction comes from backend/services/lgbm_pipeline.py (the code the API
serves), so the maps show exactly what the deployed system returns. The reference
route is drawn from the .route file for development evaluation only.

python scripts/make_maps.py [trip ...]   -> reports/maps/<trip>_lgbm_tuned.html
"""
import os
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend.services.lgbm_pipeline import evaluate, load_bundle, predict_trip  # noqa: E402
from pathsnap.io import DATA_ROOT, list_trips, load_trip  # noqa: E402
from pathsnap.labels import load_route  # noqa: E402
from pathsnap.roadnet import RoadNet  # noqa: E402

OUT = os.path.join(ROOT, "reports", "maps")
COL = dict(network="#9aa0a6", reference="#1b9e77", predicted="#d95f02", classifier="#7570b3", gps="#1f78b4")


def _latlon(coords):
    return [[c[1], c[0]] for c in coords]


def _chain(route_items):
    """Join consecutive arcs into polylines; start a new line where arcs do not touch."""
    lines, cur = [], []
    for it in route_items:
        a, b = it["coordinates"]
        if cur and cur[-1] == a:
            cur.append(b)
        else:
            if len(cur) > 1:
                lines.append(cur)
            cur = [a, b]
    if len(cur) > 1:
        lines.append(cur)
    return lines


def legend(title, ev):
    rows = "".join(f'<div><span style="display:inline-block;width:22px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{n}</div>'
                   for n, c in [("Road network (nearby)", COL["network"]), ("Reference route", COL["reference"]),
                                ("Predicted route (Viterbi)", COL["predicted"]),
                                ("Classifier-only route", COL["classifier"]), ("GPS points", COL["gps"])])
    m = ""
    if ev:
        m = (f'<div style="margin-top:6px">Top-1: classifier {ev["classifier_top1_accuracy"]:.3f} &rarr; '
             f'Viterbi {ev["viterbi_top1_accuracy"]:.3f}<br>macro F1 (Viterbi) {ev["viterbi_macro_f1"]:.3f} &middot; '
             f'continuity {ev["viterbi_route_continuity"]:.3f}</div>')
    return folium.Element(
        f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:10px 12px;'
        f'border:1px solid #bbb;border-radius:6px;font:12px sans-serif;box-shadow:0 1px 4px rgba(0,0,0,.2)">'
        f'<b>{title}</b>{rows}{m}</div>')


def make(tid, part="student_test"):
    d = Path(DATA_ROOT) / part / tid
    pred = predict_trip(d, load_bundle())
    ev = evaluate(pred, d)
    trip = load_trip(part, tid)
    rn = RoadNet(trip)
    xy = rn.project_track(trip.track)
    _, near = rn.query_radius(xy[::5], 150.0)
    near = np.unique(near)
    nodes, arcs = trip.nodes, trip.arcs
    ref = rn.canon[load_route(part, tid)]

    gps = np.array([o["gps"] for o in pred["observations"]])
    m = folium.Map(location=[gps[:, 1].mean(), gps[:, 0].mean()], zoom_start=14, tiles="cartodbpositron",
                   control_scale=True)
    fg = folium.FeatureGroup(name="Road network (within 150 m)", show=True)
    for a in near:
        folium.PolyLine(_latlon(nodes[arcs[a, :2]].tolist()), color=COL["network"], weight=1.5, opacity=0.6).add_to(fg)
    fg.add_to(m)
    fr = folium.FeatureGroup(name="Reference route (.route, evaluation only)")
    ref_items = [{"coordinates": nodes[arcs[a, :2]].tolist()} for a in ref]
    for ln in _chain(ref_items):
        folium.PolyLine(_latlon(ln), color=COL["reference"], weight=8, opacity=0.55).add_to(fr)
    fr.add_to(m)
    fp = folium.FeatureGroup(name="Predicted route (tuned LightGBM + Viterbi)")
    for ln in _chain(pred["predicted_route"]):
        folium.PolyLine(_latlon(ln), color=COL["predicted"], weight=4, opacity=0.95).add_to(fp)
    fp.add_to(m)
    fc = folium.FeatureGroup(name="Classifier-only route (no Viterbi)", show=False)
    for ln in _chain(pred["classifier_route"]):
        folium.PolyLine(_latlon(ln), color=COL["classifier"], weight=3, opacity=0.9, dash_array="6 6").add_to(fc)
    fc.add_to(m)
    fg2 = folium.FeatureGroup(name="GPS points (preprocessed)")
    for o in pred["observations"]:
        folium.CircleMarker([o["gps"][1], o["gps"][0]], radius=2.5, color=COL["gps"], fill=True, fill_opacity=0.9,
                            weight=0, tooltip=f't={o["timestamp"]:.0f}s arc={o["arc_id"]} p={o["confidence"]:.2f}"').add_to(fg2)
    fg2.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[gps[:, 1].min(), gps[:, 0].min()], [gps[:, 1].max(), gps[:, 0].max()]])
    m.get_root().html.add_child(legend(f"Student-test trip {tid}", ev))
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{tid}_lgbm_tuned.html")
    m.save(path)
    return dict(trip=tid, map=os.path.relpath(path, ROOT), **(ev or {}))


if __name__ == "__main__":
    trips = sys.argv[1:] or list_trips("student_test")
    rows = [make(t) for t in trips]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "lgbm_tuned_map_index.csv"), index=False)
    print(df[["trip", "classifier_top1_accuracy", "viterbi_top1_accuracy", "viterbi_macro_f1", "map"]].round(4)
          .to_string(index=False))
