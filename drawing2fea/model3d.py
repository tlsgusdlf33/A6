"""3D solid modeling from a 2D profile: extrusion and revolution.

The 2D triangulation is swept into prisms (extrude) or wedge rings (revolve),
and each prism is split into three tetrahedra. Prism splitting follows the
"lowest global index" rule so that neighbouring prisms choose the same diagonal
on their shared quad face, which guarantees a conforming tetrahedral mesh.
Optionally the linear tets are upgraded to 10-node quadratic tets.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mesh2d import Mesh2D

# Local corner indices of the four faces of a tetrahedron (opposite node last).
TET_FACES = np.array([[0, 2, 1, 3], [0, 1, 3, 2], [1, 2, 3, 0], [0, 3, 2, 1]])
# VTK quadratic tetra edge order for midside nodes 4..9.
TET10_EDGES = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]])


@dataclass
class Mesh3D:
    nodes: np.ndarray            # (n, 3)
    elements: np.ndarray         # (m, 4) or (m, 10)
    info: dict = field(default_factory=dict)

    @property
    def order(self) -> int:
        return 2 if self.elements.shape[1] == 10 else 1

    @property
    def corners(self) -> np.ndarray:
        return self.elements[:, :4]

    def element_volumes(self) -> np.ndarray:
        p = self.nodes[self.corners]
        return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0

    @property
    def volume(self) -> float:
        return float(self.element_volumes().sum())

    def boundary_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """Return outward-oriented boundary faces.

        ``(faces_corner (k,3), faces_full (k,3|6))`` where the full form for
        tet10 lists ``[c0, c1, c2, m01, m12, m20]``.
        """
        if "_bfaces" in self.info:
            return self.info["_bfaces"]
        el = self.elements
        local = TET_FACES[:, :3]
        faces = el[:, local].reshape(-1, 3)          # outward by construction
        key = np.sort(faces, axis=1)
        _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        bmask = counts[inv.ravel()] == 1
        bf = faces[bmask]
        if self.order == 2:
            # midside node of each face edge, taken from the owning element
            elem_id = np.repeat(np.arange(len(el)), 4)[bmask]
            face_id = np.tile(np.arange(4), len(el))[bmask]
            edge_lookup = {}
            for e_idx, (a, b) in enumerate(TET10_EDGES):
                edge_lookup[(a, b)] = 4 + e_idx
                edge_lookup[(b, a)] = 4 + e_idx
            mids = np.empty((len(bf), 3), dtype=el.dtype)
            for f in range(4):
                sel = face_id == f
                lf = local[f]
                for k in range(3):
                    mids[sel, k] = el[elem_id[sel], edge_lookup[(lf[k], lf[(k + 1) % 3])]]
            full = np.hstack([bf, mids])
        else:
            full = bf
        self.info["_bfaces"] = (bf, full)
        return bf, full


# --------------------------------------------------------------------------- #
# prism splitting
# --------------------------------------------------------------------------- #
def _prisms_to_tets(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Split prisms (bottom i,j,k / top i',j',k') into 3 tets each.

    The bottom triangle must be sorted by ascending *2D* node index; then the
    quad face diagonal always runs from the lower-index bottom node to the
    higher-index top node, which is consistent between neighbours.
    """
    i, j, k = bottom.T
    i2, j2, k2 = top.T
    return np.vstack([
        np.c_[i, j, k, k2],
        np.c_[i, j, j2, k2],
        np.c_[i, i2, j2, k2],
    ])


def _fix_tets(nodes: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Drop degenerate tets (collapsed on an axis) and make all positively oriented."""
    tets = tets[(np.sort(tets, axis=1)[:, 1:] != np.sort(tets, axis=1)[:, :-1]).all(axis=1)]
    p = nodes[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    scale = np.abs(vol).max()
    tets = tets[np.abs(vol) > 1e-10 * scale]
    vol = vol[np.abs(vol) > 1e-10 * scale]
    neg = vol < 0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return tets


def extrude(mesh2d: Mesh2D, thickness: float, h: float, layers: int | None = None) -> Mesh3D:
    """Extrude along +z from z=0 to z=thickness."""
    if thickness <= 0:
        raise ValueError("extrusion thickness must be positive")
    n_layers = layers or max(2, int(np.ceil(thickness / h - 1e-9)))
    z = np.linspace(0.0, thickness, n_layers + 1)
    n2 = len(mesh2d.points)
    nodes = np.vstack([np.c_[mesh2d.points, np.full(n2, zk)] for zk in z])
    tri = np.sort(mesh2d.triangles, axis=1)
    tets = np.vstack([_prisms_to_tets(tri + k * n2, tri + (k + 1) * n2) for k in range(n_layers)])
    tets = _fix_tets(nodes, tets)
    return Mesh3D(nodes, tets, {"operation": "extrude", "thickness": thickness, "layers": n_layers})


def revolve(mesh2d: Mesh2D, angle_deg: float, h: float, axis_x: float = 0.0, segments: int | None = None) -> Mesh3D:
    """Revolve the profile about the drawing's vertical line ``x = axis_x``.

    Drawing coordinates (x, y) are interpreted as radius r = x - axis_x and
    height z = y. The 3D axis of revolution is the global z axis.
    """
    r = mesh2d.points[:, 0] - axis_x
    zc = mesh2d.points[:, 1]
    rmax = float(r.max())
    if r.min() < -1e-9 * rmax:
        raise ValueError("revolve profile crosses the axis of revolution (x < axis_x)")
    r = np.clip(r, 0.0, None)
    full = abs(angle_deg - 360.0) < 1e-9
    ang = np.radians(angle_deg)
    n_seg = segments or max(16 if full else 4, int(np.ceil(ang * rmax / h)))
    n2 = len(r)
    on_axis = r <= 1e-9 * rmax

    n_layers = n_seg if full else n_seg + 1
    theta = np.linspace(0.0, ang, n_seg + 1)[:n_layers]
    # Map (layer, 2D node) -> 3D node, collapsing on-axis nodes to layer 0.
    idx = np.empty((n_layers, n2), dtype=np.int64)
    nodes = []
    count = 0
    for k, th in enumerate(theta):
        sel = np.ones(n2, bool) if k == 0 else ~on_axis
        ids = np.arange(count, count + sel.sum())
        idx[k, sel] = ids
        if k > 0:
            idx[k, on_axis] = idx[0, on_axis]
        nodes.append(np.c_[r[sel] * np.cos(th), r[sel] * np.sin(th), zc[sel]])
        count += sel.sum()
    nodes = np.vstack(nodes)

    tri = np.sort(mesh2d.triangles, axis=1)
    tets = []
    for k in range(n_seg):
        k2 = (k + 1) % n_layers
        tets.append(_prisms_to_tets(idx[k][tri], idx[k2][tri]))
    tets = _fix_tets(nodes, np.vstack(tets))
    return Mesh3D(nodes, tets, {"operation": "revolve", "angle_deg": angle_deg, "segments": n_seg, "axis_x": axis_x})


# --------------------------------------------------------------------------- #
# quadratic elements
# --------------------------------------------------------------------------- #
def to_quadratic(mesh: Mesh3D) -> Mesh3D:
    """Convert a tet4 mesh to tet10 by adding (shared) edge midpoint nodes."""
    if mesh.order == 2:
        return mesh
    tets = mesh.corners
    edges = np.sort(tets[:, TET10_EDGES].reshape(-1, 2), axis=1)
    uniq, inv = np.unique(edges, axis=0, return_inverse=True)
    mid_nodes = 0.5 * (mesh.nodes[uniq[:, 0]] + mesh.nodes[uniq[:, 1]])
    mid_ids = len(mesh.nodes) + inv.reshape(-1, 6)
    info = {k: v for k, v in mesh.info.items() if not k.startswith("_")}
    return Mesh3D(np.vstack([mesh.nodes, mid_nodes]), np.hstack([tets, mid_ids]), info)


def remove_unused_nodes(mesh: Mesh3D) -> Mesh3D:
    used = np.unique(mesh.elements)
    if len(used) == len(mesh.nodes):
        return mesh
    remap = -np.ones(len(mesh.nodes), dtype=np.int64)
    remap[used] = np.arange(len(used))
    info = {k: v for k, v in mesh.info.items() if not k.startswith("_")}
    return Mesh3D(mesh.nodes[used], remap[mesh.elements], info)
