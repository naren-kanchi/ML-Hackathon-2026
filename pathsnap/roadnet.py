"""Projected road network with a spatial index (inference-safe; no route access).

Projection: per-trip azimuthal-equidistant (AEQD) plane centred on the GPS track
centroid. Trips span < ~100 km, so AEQD distance distortion is far below GPS
noise, and it works anywhere on Earth (the dataset spans 5 continents, so a
single UTM zone is not an option).

Arcs are straight directed segments node[from] -> node[to].
Duplicate arcs (identical from,to) are geometrically indistinguishable; we map
each arc to a canonical id = smallest arc id with the same (from,to). Candidate
generation returns canonical ids only, and labels/metrics compare canonical ids.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from pyproj import Transformer
from shapely import STRtree, linestrings

from .io import Trip


class RoadNet:
    def __init__(self, trip: Trip):
        lon0, lat0 = np.mean(trip.track[:, 0]), np.mean(trip.track[:, 1])
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self.fwd = Transformer.from_crs(
            "EPSG:4326", f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", always_xy=True)
        self.inv = Transformer.from_crs(
            f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m", "EPSG:4326", always_xy=True)
        nx, ny = self.fwd.transform(trip.nodes[:, 0], trip.nodes[:, 1])
        self.xy = np.column_stack([nx, ny])
        self.arcs = trip.arcs
        A, N = len(trip.arcs), len(trip.nodes)
        self.A, self.N = A, N
        fr, to = trip.arcs[:, 0], trip.arcs[:, 1]
        self.fr, self.to = fr, to
        self.p0 = self.xy[fr]
        self.p1 = self.xy[to]
        d = self.p1 - self.p0
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.heading = np.arctan2(d[:, 1], d[:, 0])  # math angle, radians
        # canonical ids for duplicate (from,to)
        key = fr.astype(np.int64) * N + to
        _, first_idx, inv = np.unique(key, return_index=True, return_inverse=True)
        self.canon = first_idx[inv]
        self.is_canon = self.canon == np.arange(A)
        # reverse twin (canonical) or -1
        rkey = to.astype(np.int64) * N + fr
        pos = np.searchsorted(key[first_idx], rkey)  # keys of uniques are sorted
        uk = key[first_idx]
        pos = np.clip(pos, 0, len(uk) - 1)
        self.reverse = np.where(uk[pos] == rkey, first_idx[pos], -1)
        # successor structure over canonical arcs: arc a -> arcs leaving node to[a]
        cid = np.nonzero(self.is_canon)[0]
        order = np.argsort(fr[cid], kind="stable")
        self._out_arcs = cid[order]
        self._out_ptr = np.searchsorted(fr[cid][order], np.arange(N + 1))
        order_in = np.argsort(to[cid], kind="stable")
        self._in_arcs = cid[order_in]
        self._in_ptr = np.searchsorted(to[cid][order_in], np.arange(N + 1))
        # node graph (canonical arcs, weight=length) for shortest paths
        self.node_graph = sp.csr_matrix(
            (np.maximum(self.length[cid], 1e-3), (fr[cid], to[cid])), shape=(N, N))
        # spatial index over canonical segments
        segs = np.stack([self.p0[cid], self.p1[cid]], axis=1)
        self._cid = cid
        self.tree = STRtree(linestrings(segs))

    # ---- projection helpers
    def project_track(self, track: np.ndarray) -> np.ndarray:
        x, y = self.fwd.transform(track[:, 0], track[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self.inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])

    # ---- topology
    def successors(self, a: int) -> np.ndarray:
        n = self.to[a]
        return self._out_arcs[self._out_ptr[n]:self._out_ptr[n + 1]]

    def predecessors(self, a: int) -> np.ndarray:
        n = self.fr[a]
        return self._in_arcs[self._in_ptr[n]:self._in_ptr[n + 1]]

    def out_degree_node(self, n):
        return self._out_ptr[n + 1] - self._out_ptr[n]

    def in_degree_node(self, n):
        return self._in_ptr[n + 1] - self._in_ptr[n]

    # ---- geometry
    def project_points(self, pts: np.ndarray, arc_ids: np.ndarray):
        """Point-to-segment for paired (pts[i], arc_ids[i]).
        Returns dist, offset-along-arc (m), fraction in [0,1], foot point."""
        a = self.p0[arc_ids]
        d = self.p1[arc_ids] - a
        L2 = np.maximum((d * d).sum(1), 1e-9)
        f = np.clip(((pts - a) * d).sum(1) / L2, 0.0, 1.0)
        foot = a + f[:, None] * d
        dist = np.hypot(*(pts - foot).T)
        return dist, f * np.sqrt(L2), f, foot

    def query_radius(self, pts: np.ndarray, radius: float):
        """All canonical arcs within `radius` of each point. Returns (pt_idx, arc_id)."""
        from shapely import points
        pi, si = self.tree.query(points(pts), predicate="dwithin", distance=radius)
        return pi, self._cid[si]
