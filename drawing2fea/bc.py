"""Face selectors, boundary conditions and loads.

Selector grammar (used as ``"on"`` in boundary conditions and loads)::

    xmin | xmax | ymin | ymax | zmin | zmax   faces on the bounding-box planes
    top | bottom                              aliases of zmax / zmin
    hole:<k> | hole:all                       wall of hole k (extruded profiles and 3-view models)
    outer                                     side walls of outer loops (extrude only)
    x<10  y>=5  z=0  r<=12.5                  coordinate predicates (r = sqrt(x²+y²))
    box:x0,x1,y0,y1,z0,z1                     axis-aligned box (use * for unbounded)
    all                                       every boundary face
    A & B                                     intersection, e.g. "xmin & y>20"

A face is selected when all of its corner nodes satisfy the predicate.
"""

from __future__ import annotations

import re

import numpy as np

from .drawing import Profile2D
from .fem import Constraints, body_force_vector, face_load_vector, face_normals_areas
from .materials import Material
from .model3d import Mesh3D
from .views import HoleSpec

_CMP = re.compile(r"^\s*([xyzr])\s*(<=|>=|<|>|=)\s*([-+0-9.eE]+)\s*$")
_AXIS = {"x": 0, "y": 1, "z": 2}


def _point_segment_distance(pts: np.ndarray, loop: np.ndarray) -> np.ndarray:
    a = loop
    b = np.roll(loop, -1, axis=0)
    ab = b - a
    ap = pts[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("pij,ij->pi", ap, ab) / np.einsum("ij,ij->i", ab, ab), 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    return np.linalg.norm(pts[:, None, :] - proj, axis=2).min(axis=1)


class FaceSelector:
    def __init__(self, mesh: Mesh3D, profile: Profile2D | None = None, h: float | None = None,
                 holes: list | None = None):
        self.mesh = mesh
        self.profile = profile
        self.faces, self.faces_full = mesh.boundary_faces()
        x = mesh.nodes
        self.lo, self.hi = x.min(axis=0), x.max(axis=0)
        self.size = float(np.max(self.hi - self.lo))
        self.tol = 1e-6 * self.size
        self.h = h or self.size / 20
        if holes is None and profile is not None and mesh.info.get("operation") == "extrude":
            holes = [HoleSpec((0, 1), hl, 2, None, 1e-3 * self.h) for hl in profile.holes]
        self.holes = holes or []

    # -- node predicates ---------------------------------------------------- #
    def _node_mask(self, expr: str) -> np.ndarray:
        x = self.mesh.nodes
        e = expr.strip()
        el = e.lower()
        n = len(x)
        if el == "all":
            return np.ones(n, bool)
        if el in ("top", "bottom"):
            el = "zmax" if el == "top" else "zmin"
        if re.fullmatch(r"[xyz](min|max)", el):
            ax = _AXIS[el[0]]
            ref = self.lo[ax] if el.endswith("min") else self.hi[ax]
            return np.abs(x[:, ax] - ref) <= self.tol
        m = _CMP.match(el)
        if m:
            c, op, v = m.group(1), m.group(2), float(m.group(3))
            val = np.hypot(x[:, 0], x[:, 1]) if c == "r" else x[:, _AXIS[c]]
            t = self.tol
            return {
                "<": val < v + t, "<=": val <= v + t, ">": val > v - t, ">=": val >= v - t,
                "=": np.abs(val - v) <= t,
            }[op]
        if el.startswith("box:"):
            vals = [p.strip() for p in el[4:].split(",")]
            if len(vals) != 6:
                raise ValueError(f"box selector needs 6 values: {expr}")
            lim = [(-np.inf if v in ("*", "") else float(v)) for v in vals[0::2]], \
                  [(np.inf if v in ("*", "") else float(v)) for v in vals[1::2]]
            lo, hi = np.array(lim[0]) - self.tol, np.array(lim[1]) + self.tol
            return ((x >= lo) & (x <= hi)).all(axis=1)
        if el == "outer":
            if self.profile is None or self.mesh.info.get("operation") != "extrude":
                raise ValueError("selector 'outer' is only available for extruded models")
            return self._wall_mask([HoleSpec((0, 1), r.outer, 2, None, 1e-3 * self.h)
                                    for r in self.profile.regions])
        if el.startswith("hole:"):
            holes = self.holes
            if not holes:
                raise ValueError("the model has no holes")
            which = el[5:].strip()
            if which == "all":
                return self._wall_mask(holes)
            k = int(which)
            if not 0 <= k < len(holes):
                raise ValueError(f"hole index {k} out of range (0..{len(holes) - 1})")
            return self._wall_mask([holes[k]])
        raise ValueError(f"unknown selector '{expr}'")

    def _wall_mask(self, specs: list) -> np.ndarray:
        """Nodes on the side wall of a loop swept along an axis (hole walls)."""
        x = self.mesh.nodes
        cand = np.unique(self.faces_full)
        mask = np.zeros(len(x), bool)
        for sp in specs:
            pts = x[cand][:, list(sp.axes)]
            ok = _point_segment_distance(pts, sp.loop) <= sp.tol
            if sp.range is not None:
                t = x[cand, sp.axis]
                ok &= (t >= sp.range[0] - sp.tol - self.tol) & (t <= sp.range[1] + sp.tol + self.tol)
            mask[cand[ok]] = True
        return mask

    def node_mask(self, expr: str) -> np.ndarray:
        parts = [p for p in expr.split("&")]
        mask = self._node_mask(parts[0])
        for p in parts[1:]:
            mask &= self._node_mask(p)
        return mask

    def select(self, expr: str) -> np.ndarray:
        """Indices of boundary faces matching ``expr``."""
        mask = self.node_mask(expr)
        sel = np.flatnonzero(mask[self.faces].all(axis=1))
        if len(sel) == 0:
            raise ValueError(f"selector '{expr}' matched no boundary faces")
        return sel


# --------------------------------------------------------------------------- #
# build constraint set and load vector from a job description
# --------------------------------------------------------------------------- #
def build_constraints(selector: FaceSelector, bcs: list[dict]) -> tuple[Constraints, list[dict]]:
    dof_values: dict[int, float] = {}
    log = []
    for bc in bcs:
        kind = bc.get("type", "fixed").lower()
        faces = selector.select(bc["on"])
        nodes = np.unique(selector.faces_full[faces])
        if kind == "fixed":
            comps = {0: 0.0, 1: 0.0, 2: 0.0}
        elif kind in ("displacement", "symmetry"):
            comps = {_AXIS[c]: float(bc[c]) for c in "xyz" if c in bc}
            if kind == "symmetry":
                comps = {_AXIS[bc["normal"].lower()]: 0.0}
            if not comps:
                raise ValueError(f"displacement BC on '{bc['on']}' prescribes no component")
        else:
            raise ValueError(f"unknown boundary condition type '{kind}'")
        for c, v in comps.items():
            for d in 3 * nodes + c:
                dof_values[int(d)] = v
        log.append({"type": kind, "on": bc["on"], "faces": int(len(faces)), "nodes": int(len(nodes)),
                    "components": "".join("xyz"[c] for c in sorted(comps))})
    dofs = np.array(sorted(dof_values), dtype=np.int64)
    vals = np.array([dof_values[d] for d in dofs], dtype=float)
    return Constraints(dofs, vals), log


def build_load_vector(selector: FaceSelector, mat: Material, loads: list[dict]) -> tuple[np.ndarray, list[dict]]:
    mesh = selector.mesh
    f = np.zeros(3 * len(mesh.nodes))
    log = []
    for ld in loads:
        kind = ld.get("type", "force").lower()
        if kind == "gravity":
            acc = np.asarray(ld.get("value", [0, -9810.0, 0]), float)
            f += body_force_vector(mesh, mat, acc)
            log.append({"type": "gravity", "accel_mm_s2": acc.tolist()})
            continue
        faces = selector.select(ld["on"])
        normals, areas = face_normals_areas(mesh, selector.faces[faces])
        A = float(areas.sum())
        if kind == "force":
            F = np.asarray(ld["value"], float)
            traction = F / A
            f += face_load_vector(mesh, selector.faces_full[faces], traction)
            log.append({"type": "force", "on": ld["on"], "total_N": F.tolist(), "area_mm2": A,
                        "traction_MPa": traction.tolist()})
        elif kind == "pressure":
            p = float(ld["value"])
            f += face_load_vector(mesh, selector.faces_full[faces], -p * normals)
            log.append({"type": "pressure", "on": ld["on"], "pressure_MPa": p, "area_mm2": A,
                        "resultant_N": (-p * (normals * areas[:, None]).sum(axis=0)).tolist()})
        else:
            raise ValueError(f"unknown load type '{kind}'")
    return f, log
