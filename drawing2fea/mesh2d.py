"""Triangulation of a 2D profile with holes.

Boundary loops are resampled at the target size ``h`` (sharp corners are always
kept), the interior is filled with an equilateral lattice and the point cloud
is Delaunay-triangulated; triangles outside the material are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .drawing import Profile2D, Region


@dataclass
class Mesh2D:
    points: np.ndarray      # (n, 2)
    triangles: np.ndarray   # (m, 3), CCW

    @property
    def area(self) -> float:
        return float(triangle_areas(self.points, self.triangles).sum())


def triangle_areas(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def resample_loop(loop: np.ndarray, h: float, corner_deg: float = 20.0, min_segments: int = 24) -> np.ndarray:
    """Resample a closed polyline to spacing ~h while keeping sharp corners."""
    n = len(loop)
    nxt = np.roll(loop, -1, axis=0)
    prv = np.roll(loop, 1, axis=0)
    d_in = loop - prv
    d_out = nxt - loop
    cosang = np.einsum("ij,ij->i", d_in, d_out) / (
        np.linalg.norm(d_in, axis=1) * np.linalg.norm(d_out, axis=1) + 1e-300
    )
    corners = np.flatnonzero(cosang < np.cos(np.radians(corner_deg)))
    perimeter = float(np.linalg.norm(d_out, axis=1).sum())
    h = min(h, perimeter / min_segments)
    if len(corners) == 0:
        corners = np.array([0])

    out = []
    for ci, start in enumerate(corners):
        end = corners[(ci + 1) % len(corners)]
        idx = np.arange(start, end + (n if end <= start else 0) + 1) % n
        seg = loop[idx]
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        k = max(1, int(np.ceil(s[-1] / h - 1e-9)))
        si = np.linspace(0.0, s[-1], k + 1)[:-1]
        out.append(np.c_[np.interp(si, s, seg[:, 0]), np.interp(si, s, seg[:, 1])])
    return np.vstack(out)


def triangulate(profile: Profile2D, h: float, max_tries: int = 4) -> Mesh2D:
    """Triangulate ``profile`` with target edge length ``h``.

    If the triangulation does not reproduce the profile area (a boundary edge
    was lost in a feature thinner than ``h``) the size is reduced and retried.
    """
    for _ in range(max_tries):
        mesh, target = _triangulate_once(profile, h)
        if abs(mesh.area - target) <= 1e-6 * target:
            return mesh
        h *= 0.6
    raise RuntimeError(
        f"2D meshing failed: mesh area {mesh.area:.6g} != profile area {target:.6g}. "
        "Check the drawing for self-intersecting or overlapping loops."
    )


def _triangulate_once(profile: Profile2D, h: float) -> tuple[Mesh2D, float]:
    """One meshing attempt; returns the mesh and the area of the discretised profile."""
    # Discretised profile: the polygon the mesh boundary must reproduce exactly.
    disc = Profile2D([Region(resample_loop(r.outer, h), [resample_loop(q, h) for q in r.holes])
                      for r in profile.regions])
    loops = disc.all_loops
    bpts = np.vstack(loops)

    # Interior equilateral lattice.
    lo, hi = profile.bounds
    dy = h * np.sqrt(3) / 2
    ys = np.arange(lo[1] + dy / 2, hi[1], dy)
    rows = []
    for j, y in enumerate(ys):
        xs = np.arange(lo[0] + (h / 2 if j % 2 else h / 4), hi[0], h)
        rows.append(np.c_[xs, np.full_like(xs, y)])
    interior = np.vstack(rows) if rows else np.empty((0, 2))
    if len(interior):
        interior = interior[disc.contains(interior)]
        # Keep lattice points away from the boundary (use a dense boundary sample).
        dense = np.vstack([resample_loop(lp, h / 4, min_segments=48) for lp in profile.all_loops])
        dist, _ = cKDTree(dense).query(interior)
        interior = interior[dist > 0.55 * h]

    # A frame of helper points keeps the profile boundary off the convex hull,
    # where collinear boundary points would otherwise produce degenerate slivers.
    pad = 2.0 * h
    fx = np.linspace(lo[0] - pad, hi[0] + pad, max(2, int(np.ceil((hi[0] - lo[0]) / h)) + 5))
    fy = np.linspace(lo[1] - pad, hi[1] + pad, max(2, int(np.ceil((hi[1] - lo[1]) / h)) + 5))
    frame = np.vstack([np.c_[fx, np.full_like(fx, fy[0])], np.c_[fx, np.full_like(fx, fy[-1])],
                       np.c_[np.full_like(fy[1:-1], fx[0]), fy[1:-1]], np.c_[np.full_like(fy[1:-1], fx[-1]), fy[1:-1]]])
    pts = np.vstack([bpts, interior, frame])
    tri = Delaunay(pts).simplices
    area = triangle_areas(pts, tri)
    tri[area < 0] = tri[area < 0][:, [0, 2, 1]]
    area = np.abs(area)

    centroid = pts[tri].mean(axis=1)
    keep = disc.contains(centroid) & (area > 1e-10 * h * h)
    # Reject triangles whose edges leave the material (narrow concave notches).
    for a, b in ((0, 1), (1, 2), (2, 0)):
        mid = 0.5 * (pts[tri[:, a]] + pts[tri[:, b]])
        # nudge toward the centroid: midpoints of boundary edges lie exactly on the boundary
        keep &= disc.contains(mid + 1e-3 * (centroid - mid))
    tri = tri[keep]

    # Compact node numbering.
    used = np.unique(tri)
    remap = -np.ones(len(pts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    pts, tri = pts[used], remap[tri]

    # Light Laplacian smoothing of interior nodes improves element quality.
    is_b = used < len(bpts)
    nbr_edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    nbr_edges = np.vstack([nbr_edges, nbr_edges[:, ::-1]])
    deg = np.bincount(nbr_edges[:, 0], minlength=len(pts)).astype(float)
    for _ in range(3):
        acc = np.zeros_like(pts)
        np.add.at(acc, nbr_edges[:, 0], pts[nbr_edges[:, 1]])
        new = acc / deg[:, None]
        cand = pts.copy()
        cand[~is_b] = new[~is_b]
        if (triangle_areas(cand, tri) > 0).all():
            pts = cand
    return Mesh2D(points=pts, triangles=tri), disc.area
