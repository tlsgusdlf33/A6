"""3D reconstruction from a three-view orthographic drawing (front / top / side).

Coordinate system of the reconstructed part::

    x = width (left -> right in the front view)
    y = depth (front face y = 0 -> back)
    z = height (bottom z = 0 -> top)

    front view  shows (x, z)     top view  shows (x, y)     side view  shows (y, z)

Reconstruction steps
--------------------
1. **View separation** - profile entities are grouped into spatial clusters
   (or taken from layers named FRONT / TOP / SIDE). The front view is the
   cluster that is horizontally aligned with one view (the side view) and
   vertically aligned with another (the top view).
2. **Projection convention** - top view above the front view = third-angle
   (KS/ASME), below = first-angle (ISO-E); ``PROJECTION=FIRST|THIRD`` overrides.
   It decides on which side of the side view the front face lies.
3. **Base solid (visual hull)** - every view's outline is extruded along its
   viewing direction and the three prisms are intersected.
4. **Features** - closed loops inside a view (visible inner loops, or closed
   hidden loops) are matched with lines parallel to the viewing direction at
   the loop's extremes in the other two views:

   * hidden lines  -> **cut** (through hole, blind hole, pocket, counterbore step);
     the depth range is taken from the hidden lines,
   * visible lines -> **boss**: the material in the boss height range is
     trimmed to the loop (visual-hull corners removed),
   * no evidence   -> ignored with a warning (or a through-cut when
     ``views.assume_through`` is set).
5. **Meshing** - the solid is sliced into slabs along one axis (chosen so that
   as few curved/slanted edges as possible are approximated); all slab cross
   sections share one constrained triangulation, so the stacked prisms form a
   conforming tetrahedral mesh. Edges along the slab axis are exact; curved
   edges seen in the two other views are approximated by fine steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .drawing import parse_annotation_text
from .model3d import Mesh3D, _fix_tets, _prisms_to_tets, remove_unused_nodes

AXIS_NAMES = "xyz"
VIEW_AXES = {"front": (0, 2), "top": (0, 1), "side": (1, 2)}
VIEW_NORMAL = {"front": 1, "top": 2, "side": 0}
ROLE_ORDER = ("top", "front", "side")

_VIEW_LAYER_KEYS = {
    "front": ("FRONT", "정면"),
    "top": ("TOP", "PLAN", "평면"),
    "side": ("SIDE", "RIGHT", "LEFT", "측면"),
}
_IGNORED_LAYERS = ("DIM", "CENTER", "CENTRE", "ANNO", "TEXT", "AXIS", "HATCH", "BORDER", "FRAME", "TITLE",
                   "중심", "치수")
_IGNORED_LTYPES = ("CENTER", "CENTRE", "DASHDOT", "PHANTOM", "CHAIN", "DIVIDE", "BORDER")
_HIDDEN_KEYS = ("HIDDEN", "HID", "DASHED", "숨은")


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Stroke:
    pts: np.ndarray          # (n, 2) drawing coordinates
    hidden: bool
    layer: str

    @property
    def bbox(self):
        return self.pts.min(axis=0), self.pts.max(axis=0)


@dataclass
class View:
    role: str
    visible: list[np.ndarray]       # polylines in view-local part coordinates
    hidden: list[np.ndarray]
    silhouette: object = None       # shapely geometry
    drawing_bbox: tuple = ()

    @property
    def axes(self) -> tuple[int, int]:
        return VIEW_AXES[self.role]

    @property
    def normal(self) -> int:
        return VIEW_NORMAL[self.role]


@dataclass
class Feature:
    view: str
    kind: str                # "cut" | "boss" | "ignored"
    polygon: object          # shapely Polygon of the loop (view-local coords)
    axis: int                # direction of the feature (viewing direction of its view)
    range: tuple[float, float]
    evidence: str
    removed: object = None   # shapely geometry actually subtracted (loop, or bbox - loop for bosses)
    through: bool = False

    @property
    def axes(self) -> tuple[int, int]:
        return VIEW_AXES[self.view]

    def describe(self) -> str:
        c = self.polygon.centroid
        a, b = self.axes
        lo, hi = self.range
        if self.kind == "cut":
            what = "through hole/cut" if self.through else "blind hole/pocket"
        elif self.kind == "boss":
            what = "boss"
        else:
            what = "ignored loop"
        return (f"{what} from {self.view} view at {AXIS_NAMES[a]}={c.x:.4g}, {AXIS_NAMES[b]}={c.y:.4g}, "
                f"{AXIS_NAMES[self.axis]} {lo:.4g}..{hi:.4g} ({self.evidence})")


@dataclass
class ThreeViewDrawing:
    views: dict[str, View]
    size: np.ndarray                     # (W, D, H)
    projection: str
    side_placement: str
    features: list[Feature] = field(default_factory=list)
    annotations: dict[str, str] = field(default_factory=dict)
    source: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def cuts(self) -> list[Feature]:
        return [f for f in self.features if f.kind == "cut"]

    def summary(self) -> dict:
        return {
            "mode": "three_views",
            "source": self.source,
            "projection": self.projection,
            "side_view": self.side_placement,
            "size_mm": self.size.tolist(),
            "views": {r: {"visible_lines": len(v.visible), "hidden_lines": len(v.hidden),
                          "outline_area_mm2": float(v.silhouette.area)} for r, v in self.views.items()},
            "features": [f.describe() for f in self.features],
            "holes": len(self.cuts),
            "annotations": dict(self.annotations),
        }


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def _linetype(e, doc) -> str:
    lt = (e.dxf.get("linetype", "BYLAYER") or "BYLAYER").upper()
    if lt in ("BYLAYER", "BYBLOCK"):
        try:
            lt = (doc.layers.get(e.dxf.layer).dxf.linetype or "").upper()
        except Exception:
            lt = ""
    return lt


def read_strokes(path: str | Path) -> tuple[list[Stroke], dict[str, str]]:
    """All profile strokes (visible and hidden) and KEY=VALUE annotations of a DXF."""
    import ezdxf
    from ezdxf import path as dxfpath

    doc = ezdxf.readfile(str(path))
    annotations: dict[str, str] = {}
    paths = []
    for e in doc.modelspace():
        kind = e.dxftype()
        if kind in ("TEXT", "MTEXT"):
            text = e.plain_text() if kind == "MTEXT" else e.dxf.text
            for k, v in parse_annotation_text(text).items():
                annotations[k] = f"{annotations[k]};{v}" if k in annotations else v
            continue
        if kind not in ("LINE", "ARC", "CIRCLE", "ELLIPSE", "LWPOLYLINE", "POLYLINE", "SPLINE"):
            continue
        layer = e.dxf.layer
        up = layer.upper()
        lt = _linetype(e, doc)
        if any(k in up for k in _IGNORED_LAYERS) or any(k in lt for k in _IGNORED_LTYPES):
            continue
        hidden = any(k in up for k in _HIDDEN_KEYS) or any(k in lt for k in _HIDDEN_KEYS)
        try:
            paths.append((dxfpath.make_path(e), hidden, layer))
        except (TypeError, ValueError):
            continue
    if not paths:
        return [], annotations
    ext = np.array([(v.x, v.y) for p, _, _ in paths for v in p.control_vertices()] or [(0.0, 0.0)])
    size = float(np.ptp(ext, axis=0).max()) or 1.0
    strokes = []
    for p, hidden, layer in paths:
        pts = np.array([(v.x, v.y) for v in p.flattening(distance=2e-4 * size)], dtype=float)
        if len(pts) >= 2 and np.ptp(pts, axis=0).max() > 1e-9 * size:
            strokes.append(Stroke(pts, hidden, layer))
    return strokes, annotations


def _cluster(strokes: list[Stroke], gap: float) -> list[list[int]]:
    lo = np.array([s.bbox[0] for s in strokes])
    hi = np.array([s.bbox[1] for s in strokes])
    n = len(strokes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    touch = ((lo[:, None, :] <= hi[None, :, :] + gap) & (hi[:, None, :] >= lo[None, :, :] - gap)).all(axis=2)
    for i, j in zip(*np.nonzero(np.triu(touch, 1))):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _bbox_of(strokes: list[Stroke], idx: list[int]) -> np.ndarray:
    pts = np.vstack([strokes[i].pts for i in idx])
    return np.r_[pts.min(axis=0), pts.max(axis=0)]  # umin, vmin, umax, vmax


def find_view_groups(strokes: list[Stroke]) -> list[list[int]]:
    """Spatial groups of strokes, largest first (drawing frames are discarded)."""
    if not strokes:
        return []
    pts = np.vstack([s.pts for s in strokes])
    glo, ghi = pts.min(axis=0), pts.max(axis=0)
    size = float(np.max(ghi - glo)) or 1.0
    active = list(range(len(strokes)))
    for _ in range(3):
        groups = _cluster([strokes[i] for i in active], gap=2e-3 * size)
        groups = [[active[i] for i in g] for g in groups]
        if len(_significant(strokes, groups)) >= 3:
            break
        # Remove a drawing frame: strokes spanning (almost) the whole sheet.
        span = ghi - glo
        keep = []
        for i in active:
            lo, hi = strokes[i].bbox
            if ((hi - lo) >= 0.9 * span).any():
                continue
            keep.append(i)
        if len(keep) == len(active):
            break
        active = keep
        if active:
            pts = np.vstack([strokes[i].pts for i in active])
            glo, ghi = pts.min(axis=0), pts.max(axis=0)
    return _significant(strokes, groups)


def _significant(strokes, groups):
    boxes = [_bbox_of(strokes, g) for g in groups]
    areas = [max((b[2] - b[0]), 1e-12) * max((b[3] - b[1]), 1e-12) for b in boxes]
    order = np.argsort(areas)[::-1]
    if not len(order):
        return []
    biggest = areas[order[0]]
    return [groups[i] for i in order if areas[i] >= 0.02 * biggest]


def _iou1d(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def assign_roles(strokes: list[Stroke], groups: list[list[int]]) -> dict[str, list[int]]:
    """Return {"front": idx, "top": idx, "side": idx} from layer names or layout."""
    by_layer: dict[str, list[int]] = {}
    for i, s in enumerate(strokes):
        up = s.layer.upper()
        for role, keys in _VIEW_LAYER_KEYS.items():
            if any(k in up for k in keys):
                by_layer.setdefault(role, []).append(i)
                break
    if len(by_layer) == 3:
        return by_layer

    if len(groups) < 3:
        raise ValueError(f"three views (front/top/side) expected, found {len(groups)} separate drawing group(s)")
    cand = groups[:4]
    boxes = [_bbox_of(strokes, g) for g in cand]
    best, best_score = None, -1.0
    for f in range(len(cand)):
        for t in range(len(cand)):
            for s in range(len(cand)):
                if len({f, t, s}) < 3:
                    continue
                F, T, S = boxes[f], boxes[t], boxes[s]
                # top: same u-range, disjoint in v; side: same v-range, disjoint in u
                if not (T[1] >= F[3] - 1e-9 or T[3] <= F[1] + 1e-9):
                    continue
                if not (S[0] >= F[2] - 1e-9 or S[2] <= F[0] + 1e-9):
                    continue
                score = _iou1d(F[0], F[2], T[0], T[2]) + _iou1d(F[1], F[3], S[1], S[3])
                if score > best_score:
                    best, best_score = (f, t, s), score
    if best is None or best_score < 1.2:
        raise ValueError("could not identify front / top / side views from the layout: the top view must be "
                         "vertically aligned with the front view and the side view horizontally aligned "
                         "(or put the views on layers named FRONT, TOP and SIDE)")
    return {"front": cand[best[0]], "top": cand[best[1]], "side": cand[best[2]]}


def _parse_projection(text: str | None) -> str | None:
    if not text:
        return None
    t = text.strip().upper()
    if t in ("FIRST", "1", "1ST", "FIRST_ANGLE", "ISO-E", "ISO_E", "제1각법", "1각법"):
        return "first"
    if t in ("THIRD", "3", "3RD", "THIRD_ANGLE", "ASME", "KS", "제3각법", "3각법"):
        return "third"
    raise ValueError(f"unknown projection '{text}' (use FIRST or THIRD)")


# --------------------------------------------------------------------------- #
# geometry helpers (shapely)
# --------------------------------------------------------------------------- #
def _faces(polylines: list[np.ndarray]):
    from shapely.geometry import LineString
    from shapely.ops import polygonize, unary_union

    if not polylines:
        return []
    noded = unary_union([LineString(p) for p in polylines])
    return [f for f in polygonize(noded) if f.area > 0]


def _union(geoms):
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    geoms = [g for g in geoms if g is not None and not g.is_empty]
    return unary_union(geoms) if geoms else Polygon()


def _segments(polylines: list[np.ndarray]) -> np.ndarray:
    segs = [np.stack([p[:-1], p[1:]], axis=1) for p in polylines if len(p) >= 2]
    return np.vstack(segs) if segs else np.empty((0, 2, 2))


def _merge_intervals(iv: list[tuple[float, float]], tol: float) -> list[tuple[float, float]]:
    iv = sorted(iv)
    out: list[list[float]] = []
    for a, b in iv:
        if out and a <= out[-1][1] + tol:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _slice(geom, fix_index: int, t: float, big: float) -> list[tuple[float, float]]:
    """Intervals of ``geom`` along the free coordinate on the line coord[fix_index] = t."""
    from shapely.geometry import LineString

    if geom is None or geom.is_empty:
        return []
    line = LineString([(t, -big), (t, big)]) if fix_index == 0 else LineString([(-big, t), (big, t)])
    inter = geom.intersection(line)
    out = []
    stack = [inter]
    while stack:
        g = stack.pop()
        if hasattr(g, "geoms"):
            stack.extend(g.geoms)
        elif g.geom_type == "LineString" and g.length > 0:
            c = np.asarray(g.coords)[:, 1 - fix_index]
            out.append((float(c.min()), float(c.max())))
    return _merge_intervals(out, 1e-12)


# --------------------------------------------------------------------------- #
# loading + feature recognition
# --------------------------------------------------------------------------- #
def load_three_views(path: str | Path, projection: str | None = None, assume_through: bool = False,
                     strokes_annotations=None) -> ThreeViewDrawing:
    strokes, ann = strokes_annotations or read_strokes(path)
    if not strokes:
        raise ValueError(f"{path}: no drawing geometry found")
    groups = find_view_groups(strokes)
    roles = assign_roles(strokes, groups)
    warnings: list[str] = []

    boxes = {r: _bbox_of(strokes, idx) for r, idx in roles.items()}
    F, T, S = boxes["front"], boxes["top"], boxes["side"]
    top_above = (T[1] + T[3]) / 2 > (F[1] + F[3]) / 2
    side_right = (S[0] + S[2]) / 2 > (F[0] + F[2]) / 2
    proj = _parse_projection(projection or ann.get("PROJECTION") or ann.get("투상법"))
    if proj is None:
        proj = "third" if top_above else "first"
    # The edge of the side view nearest to the front view is the front face in
    # third-angle projection and the back face in first-angle projection.
    front_near = proj == "third"
    front_at_left = side_right == front_near

    def mapper(role):
        umin, vmin, umax, vmax = boxes[role]
        if role in ("front", "top"):
            return lambda p: np.c_[p[:, 0] - umin, p[:, 1] - vmin]
        if front_at_left:
            return lambda p: np.c_[p[:, 0] - umin, p[:, 1] - vmin]
        return lambda p: np.c_[umax - p[:, 0], p[:, 1] - vmin]

    views = {}
    for role in ROLE_ORDER:
        m = mapper(role)
        vis = [m(strokes[i].pts) for i in roles[role] if not strokes[i].hidden]
        hid = [m(strokes[i].pts) for i in roles[role] if strokes[i].hidden]
        if not vis:
            raise ValueError(f"the {role} view has no visible outline")
        v = View(role, vis, hid, drawing_bbox=tuple(boxes[role]))
        v.silhouette = _union(_faces(vis))
        if v.silhouette.is_empty:
            raise ValueError(f"the {role} view has no closed outline")
        views[role] = v

    W = F[2] - F[0]
    H = F[3] - F[1]
    D = T[3] - T[1]
    for name, a, b in (("width (front vs top)", W, T[2] - T[0]), ("height (front vs side)", H, S[3] - S[1]),
                       ("depth (top vs side)", D, S[2] - S[0])):
        if abs(a - b) > 1e-3 * max(W, H, D):
            warnings.append(f"view sizes disagree in {name}: {a:.6g} vs {b:.6g} mm")

    tv = ThreeViewDrawing(views=views, size=np.array([W, D, H]), projection=proj,
                          side_placement="right of front" if side_right else "left of front",
                          annotations=ann, source=str(path), warnings=warnings)
    tv.features = detect_features(tv, assume_through)
    for f in tv.features:
        if f.kind == "ignored":
            tv.warnings.append(f"inner loop ignored - {f.describe()}")
    return tv


def _loops_of_view(v: View) -> list:
    from shapely.geometry import Polygon

    loops = []
    for face in _faces(v.visible):
        for ring in face.interiors:
            loops.append(Polygon(ring))
    for face in _faces(v.hidden):
        loops.append(Polygon(face.exterior))
    # merge identical loops and drop loops that are unions of other faces' rings twice
    out = []
    for p in sorted(loops, key=lambda p: (round(p.centroid.x, 6), round(p.centroid.y, 6), -p.area)):
        if any(abs(p.area - q.area) <= 1e-6 * max(p.area, 1e-12) and p.symmetric_difference(q).area
               <= 1e-6 * p.area for q in out):
            continue
        out.append(p)
    return out


def detect_features(tv: ThreeViewDrawing, assume_through: bool = False) -> list[Feature]:
    size = float(tv.size.max())
    tol = 2e-3 * size
    feats: list[Feature] = []
    for role in ROLE_ORDER:
        v = tv.views[role]
        a = v.normal
        extent = tv.size[a]
        for poly in _loops_of_view(v):
            best = {"hidden": None, "visible": None}
            for other in ROLE_ORDER:
                w = tv.views[other]
                if other == role or a not in w.axes:
                    continue
                ia = w.axes.index(a)          # local index of the feature axis in view w
                ic = 1 - ia                   # local index of the shared coordinate
                c = w.axes[ic]
                pb = poly.bounds              # (minx, miny, maxx, maxy) in v-local coords
                k = v.axes.index(c)
                c0, c1 = pb[k], pb[k + 2]
                for kind, lines in (("hidden", w.hidden), ("visible", w.visible)):
                    seg = _segments(lines)
                    if not len(seg):
                        continue
                    par = np.abs(seg[:, 1, ic] - seg[:, 0, ic]) <= tol * 0.5
                    par &= np.abs(seg[:, 1, ia] - seg[:, 0, ia]) > tol
                    pos = seg[:, 0, ic]
                    ranges = {}
                    for name, cv in (("c0", c0), ("c1", c1)):
                        sel = par & (np.abs(pos - cv) <= tol)
                        ranges[name] = _merge_intervals(
                            [(min(s[0, ia], s[1, ia]), max(s[0, ia], s[1, ia])) for s in seg[sel]], tol)
                    for r0 in ranges["c0"]:
                        for r1 in ranges["c1"]:
                            lo, hi = max(r0[0], r1[0]), min(r0[1], r1[1])
                            if hi - lo > tol:
                                cur = best[kind]
                                if cur is None or hi - lo > cur[1] - cur[0]:
                                    best[kind] = (lo, hi, other)
            if best["hidden"] is not None:
                lo, hi, src = best["hidden"]
                feats.append(Feature(role, "cut", poly, a, (lo, hi), f"hidden lines in {src} view", removed=poly))
            elif best["visible"] is not None:
                lo, hi, src = best["visible"]
                from shapely.geometry import box

                carve = box(*poly.bounds).difference(poly)
                feats.append(Feature(role, "boss", poly, a, (lo, hi), f"visible lines in {src} view",
                                     removed=carve if carve.area > 1e-9 * poly.area else None))
            elif assume_through:
                feats.append(Feature(role, "cut", poly, a, (0.0, extent),
                                     "no hidden lines: assumed through", removed=poly))
            else:
                feats.append(Feature(role, "ignored", poly, a, (0.0, extent),
                                     "no matching lines in the other views"))
    for f in feats:
        if f.kind == "cut":
            f.through = _cuts_through(tv, f, tol)
    return feats


def _cuts_through(tv: ThreeViewDrawing, f: Feature, tol: float) -> bool:
    """True when the cut spans all material along its axis at the loop centre."""
    c = f.polygon.representative_point()
    pos = dict(zip(f.axes, (c.x, c.y)))
    big = 10.0 * float(tv.size.max()) + 1.0
    material = [(0.0, float(tv.size[f.axis]))]
    for role, w in tv.views.items():
        if f.axis not in w.axes or role == f.view:
            continue
        other = w.axes[1 - w.axes.index(f.axis)]
        iv = _slice(w.silhouette, w.axes.index(other), pos[other], big)
        material = [(max(a, b0), min(b, b1)) for a, b in material for b0, b1 in iv if min(b, b1) > max(a, b0)]
    lo, hi = f.range
    for a, b in material:
        if b > lo + tol and a < hi - tol:  # the material interval the cut lies in
            return lo <= a + tol and hi >= b - tol
    return False


# --------------------------------------------------------------------------- #
# solid evaluation and meshing
# --------------------------------------------------------------------------- #
def _exact_coords(tv: ThreeViewDrawing) -> list[np.ndarray]:
    """Per axis: coordinates of edges that are parallel to that axis's normal planes."""
    size = float(tv.size.max())
    vals: list[list[float]] = [[0.0, float(tv.size[i])] for i in range(3)]
    for v in tv.views.values():
        geoms = [v.silhouette] + [f.removed for f in tv.features if f.view == v.role and f.removed is not None]
        for g in geoms:
            for ring in _rings(g):
                c = np.asarray(ring.coords)
                d = np.diff(c, axis=0)
                for k in range(2):  # edges with constant local coordinate k
                    sel = np.abs(d[:, k]) <= 1e-9 * size
                    vals[v.axes[k]].extend(c[:-1][sel, k].tolist())
    for f in tv.features:
        if f.kind != "ignored" and f.removed is not None:
            vals[f.axis].extend(f.range)
    return [np.unique(np.round(np.asarray(v), 9)) for v in vals]


def _rings(g):
    if g is None or g.is_empty:
        return []
    if hasattr(g, "geoms"):
        return [r for p in g.geoms for r in _rings(p)]
    if g.geom_type == "Polygon":
        return [g.exterior, *g.interiors]
    return []


def staircase_length(tv: ThreeViewDrawing, axis: int) -> float:
    """Length of outline/feature edges that would be approximated by steps for a slab axis."""
    size = float(tv.size.max())
    total = 0.0
    for v in tv.views.values():
        if axis not in v.axes:
            continue
        geoms = [v.silhouette] + [f.removed for f in tv.features if f.view == v.role and f.removed is not None]
        for g in geoms:
            for ring in _rings(g):
                d = np.diff(np.asarray(ring.coords), axis=0)
                ok = (np.abs(d) <= 1e-9 * size).any(axis=1)
                total += float(np.linalg.norm(d[~ok], axis=1).sum())
    return total


def choose_axis(tv: ThreeViewDrawing) -> int:
    scores = [staircase_length(tv, a) for a in range(3)]
    # prefer z, then y, then x on ties
    return min((2, 1, 0), key=lambda a: (round(scores[a], 6), (2, 1, 0).index(a)))


def cross_section(tv: ThreeViewDrawing, axis: int, t: float, snap=None):
    """Material cross-section at coordinate ``t`` along ``axis`` (base-view coordinates)."""
    from shapely.geometry import box

    big = 10.0 * float(tv.size.max()) + 1.0
    base_role = next(r for r, v in tv.views.items() if v.normal == axis)
    b0, b1 = VIEW_AXES[base_role]

    def rect(c_axis, lo, hi, alo=-big, ahi=big):
        if c_axis == b0:
            return box(lo, alo, hi, ahi)
        return box(alo, lo, ahi, hi)

    def intervals(geom, view_axes, fix_axis):
        iv = _slice(geom, view_axes.index(fix_axis), t, big)
        if snap is not None:
            c_axis = view_axes[1 - view_axes.index(fix_axis)]
            iv = [(snap(c_axis, lo), snap(c_axis, hi)) for lo, hi in iv]
            iv = [(lo, hi) for lo, hi in iv if hi > lo]
        return iv

    region = tv.views[base_role].silhouette
    for role, v in tv.views.items():
        if role == base_role:
            continue
        c_axis = v.axes[1 - v.axes.index(axis)]
        strips = [rect(c_axis, lo, hi) for lo, hi in intervals(v.silhouette, v.axes, axis)]
        region = region.intersection(_union(strips))
        if region.is_empty:
            return region
    for f in tv.features:
        if f.removed is None or f.kind == "ignored":
            continue
        lo, hi = f.range
        if f.axis == axis:
            if lo < t < hi:
                region = region.difference(f.removed)
        else:
            c_axis = f.axes[1 - f.axes.index(axis)]
            cut = [rect(c_axis, a, b, alo=lo, ahi=hi) for a, b in intervals(f.removed, f.axes, axis)]
            if cut:
                region = region.difference(_union(cut))
    return region


@dataclass
class HoleSpec:
    """A hole wall for face selection: loop in the (axes) plane swept over ``range`` along ``axis``."""

    axes: tuple[int, int]
    loop: np.ndarray
    axis: int
    range: tuple[float, float] | None
    tol: float


def build_view_mesh(tv: ThreeViewDrawing, h: float, axis: int | None = None) -> tuple[Mesh3D, dict]:
    """Conforming tet4 mesh of the reconstructed solid."""
    import shapely
    import triangle as tr
    from shapely.geometry import Polygon

    axis = choose_axis(tv) if axis is None else axis
    base_role = next(r for r, v in tv.views.items() if v.normal == axis)
    b0, b1 = VIEW_AXES[base_role]
    L = float(tv.size[axis])
    size = float(tv.size.max())
    exact = _exact_coords(tv)
    q = h / 4.0  # snapping grid for stepped (approximated) boundaries

    def snap(c_axis, val):
        ex = exact[c_axis]
        j = np.searchsorted(ex, val)
        for k in (j - 1, j):
            if 0 <= k < len(ex) and abs(ex[k] - val) <= 0.5 * q:
                return float(ex[k])
        return float(np.clip(round(val / q) * q, 0.0, tv.size[c_axis]))

    def same(g1, g2):
        return g1.symmetric_difference(g2).area <= 1e-9 * max(g1.area, g2.area, 1e-12) + 1e-12

    brk = exact[axis]
    brk = brk[(brk >= 0) & (brk <= L)]
    levels: list[float] = [float(brk[0])]
    slabs = []  # (geometry)
    stepped = False
    for t0, t1 in zip(brk[:-1], brk[1:]):
        if t1 - t0 <= 1e-9 * size:
            continue
        samples = [cross_section(tv, axis, t0 + f * (t1 - t0)) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        if all(same(samples[0], s) for s in samples[1:]):
            n = max(1, int(math.ceil((t1 - t0) / h - 1e-9)))
            geoms = [samples[2]] * n
        else:
            stepped = True
            n = max(2, int(math.ceil((t1 - t0) / (0.5 * h) - 1e-9)))
            geoms = [cross_section(tv, axis, t0 + (k + 0.5) * (t1 - t0) / n, snap) for k in range(n)]
        for k, g in enumerate(geoms):
            levels.append(float(t0 + (k + 1) * (t1 - t0) / n))
            slabs.append(g)

    distinct: list = []
    slab_id = []
    for g in slabs:
        for i, d in enumerate(distinct):
            if same(g, d):
                slab_id.append(i)
                break
        else:
            distinct.append(g)
            slab_id.append(len(distinct) - 1)
    nonempty = [g for g in distinct if not g.is_empty]
    if not nonempty:
        raise ValueError("the reconstructed solid is empty - check that the three views are consistent")
    U = _union(nonempty)

    # Constrained quality triangulation honouring every slab boundary.
    lines = _union([g.boundary for g in nonempty])
    key: dict[tuple, int] = {}
    verts: list[tuple[float, float]] = []
    segs = []

    def vid(p):
        k = (round(p[0] / size, 9), round(p[1] / size, 9))
        if k not in key:
            key[k] = len(verts)
            verts.append((float(p[0]), float(p[1])))
        return key[k]

    stack = [lines]
    while stack:
        g = stack.pop()
        if hasattr(g, "geoms"):
            stack.extend(g.geoms)
            continue
        c = np.asarray(g.coords)
        for p0, p1 in zip(c[:-1], c[1:]):
            i, j = vid(p0), vid(p1)
            if i != j:
                segs.append((i, j))
    holes = []
    for poly in getattr(U, "geoms", [U]):
        for ring in poly.interiors:
            gap = Polygon(ring).difference(U)
            if not gap.is_empty:
                pt = gap.representative_point()
                holes.append((pt.x, pt.y))
    data = {"vertices": np.array(verts), "segments": np.array(segs, dtype=np.int32)}
    if holes:
        data["holes"] = np.array(holes)
    amax = 0.5 * h * h
    res = tr.triangulate(data, f"pq28a{amax:.12g}Q")
    pts2 = res["vertices"]
    tri = res["triangles"]
    cen = pts2[tri].mean(axis=1)

    member = []
    for g in distinct:
        if g.is_empty:
            member.append(np.zeros(len(tri), bool))
        else:
            shapely.prepare(g)
            member.append(shapely.contains_xy(g, cen[:, 0], cen[:, 1]))

    n2 = len(pts2)
    nodes = np.zeros(((len(levels)) * n2, 3))
    for k, t in enumerate(levels):
        blk = nodes[k * n2:(k + 1) * n2]
        blk[:, b0] = pts2[:, 0]
        blk[:, b1] = pts2[:, 1]
        blk[:, axis] = t
    tri_sorted = np.sort(tri, axis=1)
    tets = []
    expected = 0.0
    for k, sid in enumerate(slab_id):
        sel = tri_sorted[member[sid]]
        if len(sel):
            tets.append(_prisms_to_tets(sel + k * n2, sel + (k + 1) * n2))
        expected += distinct[sid].area * (levels[k + 1] - levels[k])
    if not tets:
        raise ValueError("the reconstructed solid is empty")
    tets = _fix_tets(nodes, np.vstack(tets))
    mesh = remove_unused_nodes(Mesh3D(nodes, tets, {"operation": "views"}))
    info = {
        "slab_axis": AXIS_NAMES[axis],
        "slabs": len(slabs),
        "stepped_regions": stepped,
        "expected_volume_mm3": expected,
        "step_size_mm": q if stepped else None,
    }
    mesh.info.update(info)
    return mesh, info


def hole_specs(tv: ThreeViewDrawing, slab_axis: int, h: float) -> list[HoleSpec]:
    """Hole walls of all cut features (numbered as ``hole:k``)."""
    specs = []
    for f in tv.cuts:
        tol = 1e-3 * h if f.axis == slab_axis else 0.6 * h
        loop = np.asarray(f.polygon.exterior.coords)[:-1]
        specs.append(HoleSpec(f.axes, loop, f.axis, f.range, tol))
    return specs


def looks_like_three_views(strokes: list[Stroke]) -> bool:
    try:
        groups = find_view_groups(strokes)
        assign_roles(strokes, groups)
        return True
    except ValueError:
        return False

