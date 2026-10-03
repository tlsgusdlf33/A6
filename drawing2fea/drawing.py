"""Reading 2D drawings (DXF or JSON) into closed profile loops.

A drawing is turned into one or more *regions*: an outer boundary loop plus
any number of hole loops. Loops can be drawn as closed polylines/circles or as
chains of loose LINE / ARC / SPLINE entities, which are joined end-to-end.

Text entities of the form ``KEY=VALUE`` (e.g. ``THICKNESS=10``,
``MATERIAL=AL6061``, ``FIX=xmin``) are collected as *annotations* so that a
drawing alone can fully describe an analysis job.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Layers whose geometry is never part of the solid profile.
DEFAULT_IGNORED_LAYER_PATTERNS = ("DIM", "CENTER", "CENTRE", "HIDDEN", "ANNO", "TEXT", "AXIS", "HATCH")


# --------------------------------------------------------------------------- #
# polygon helpers
# --------------------------------------------------------------------------- #
def signed_area(loop: np.ndarray) -> float:
    x, y = loop[:, 0], loop[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def point_in_loop(points: np.ndarray, loop: np.ndarray) -> np.ndarray:
    """Even-odd point-in-polygon test (vectorised over points)."""
    from matplotlib.path import Path as MplPath

    return MplPath(loop).contains_points(np.atleast_2d(points))


def _dedupe(loop: np.ndarray, tol: float) -> np.ndarray:
    """Remove consecutive duplicate points and the closing duplicate."""
    keep = [0]
    for i in range(1, len(loop)):
        if np.linalg.norm(loop[i] - loop[keep[-1]]) > tol:
            keep.append(i)
    loop = loop[keep]
    if len(loop) > 1 and np.linalg.norm(loop[0] - loop[-1]) <= tol:
        loop = loop[:-1]
    return loop


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #
@dataclass
class Region:
    """A connected planar face: one outer boundary (CCW) and holes (CW)."""

    outer: np.ndarray
    holes: list[np.ndarray] = field(default_factory=list)

    @property
    def loops(self) -> list[np.ndarray]:
        return [self.outer, *self.holes]

    @property
    def area(self) -> float:
        return abs(signed_area(self.outer)) - sum(abs(signed_area(h)) for h in self.holes)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        inside = point_in_loop(pts, self.outer)
        for h in self.holes:
            inside &= ~point_in_loop(pts, h)
        return inside


@dataclass
class Profile2D:
    regions: list[Region]
    annotations: dict[str, str] = field(default_factory=dict)
    source: str = ""

    @property
    def all_loops(self) -> list[np.ndarray]:
        return [lp for r in self.regions for lp in r.loops]

    @property
    def holes(self) -> list[np.ndarray]:
        return [h for r in self.regions for h in r.holes]

    @property
    def area(self) -> float:
        return sum(r.area for r in self.regions)

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        pts = np.vstack(self.all_loops)
        return pts.min(axis=0), pts.max(axis=0)

    def contains(self, pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        inside = np.zeros(len(pts), dtype=bool)
        for r in self.regions:
            inside |= r.contains(pts)
        return inside

    def summary(self) -> dict:
        lo, hi = self.bounds
        return {
            "source": self.source,
            "regions": len(self.regions),
            "holes": len(self.holes),
            "area_mm2": self.area,
            "bbox_min": lo.tolist(),
            "bbox_max": hi.tolist(),
            "annotations": dict(self.annotations),
        }


# --------------------------------------------------------------------------- #
# loop assembly
# --------------------------------------------------------------------------- #
def chain_segments(segments: list[np.ndarray], tol: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Join open polylines end-to-end into closed loops.

    Returns ``(closed_loops, leftovers)``; leftovers are chains that never closed.
    """
    pool = [np.asarray(s, dtype=float) for s in segments if len(s) >= 2]
    loops, leftovers = [], []
    while pool:
        chain = [pool.pop(0)]
        while True:
            start, end = chain[0][0], chain[-1][-1]
            if len(chain) > 1 or len(chain[0]) > 2:
                if np.linalg.norm(end - start) <= tol:
                    loops.append(np.vstack([c[:-1] if i < len(chain) - 1 else c for i, c in enumerate(chain)]))
                    break
            for k, seg in enumerate(pool):
                if np.linalg.norm(seg[0] - end) <= tol:
                    chain.append(pool.pop(k))
                    break
                if np.linalg.norm(seg[-1] - end) <= tol:
                    chain.append(pool.pop(k)[::-1])
                    break
            else:
                leftovers.append(np.vstack(chain))
                break
    return loops, leftovers


def build_regions(loops: list[np.ndarray], tol: float = 1e-9) -> list[Region]:
    """Classify loops by nesting depth: even depth = outer boundary, odd = hole."""
    loops = [_dedupe(np.asarray(lp, dtype=float), tol) for lp in loops]
    loops = [lp for lp in loops if len(lp) >= 3 and abs(signed_area(lp)) > tol]
    if not loops:
        raise ValueError("no closed profile loops were found in the drawing")
    # Sort by area so that containers come first.
    loops.sort(key=lambda lp: -abs(signed_area(lp)))
    parents: list[int | None] = []
    for i, lp in enumerate(loops):
        # Parent = smallest earlier loop that contains most of this loop's vertices.
        parent = None
        for j in range(i - 1, -1, -1):
            if point_in_loop(lp, loops[j]).mean() > 0.5:
                parent = j
                break
        parents.append(parent)

    def depth(i: int) -> int:
        d = 0
        while parents[i] is not None:
            i = parents[i]
            d += 1
        return d

    regions: dict[int, Region] = {}
    for i, lp in enumerate(loops):
        if depth(i) % 2 == 0:
            outer = lp if signed_area(lp) > 0 else lp[::-1]
            regions[i] = Region(outer=outer.copy())
    for i, lp in enumerate(loops):
        if depth(i) % 2 == 1:
            hole = lp if signed_area(lp) < 0 else lp[::-1]
            regions[parents[i]].holes.append(hole.copy())
    out = list(regions.values())
    # Deterministic numbering: regions and holes left->right, then bottom->top.
    for r in out:
        r.holes.sort(key=lambda h: (round(float(h[:, 0].mean()), 6), round(float(h[:, 1].mean()), 6)))
    out.sort(key=lambda r: (round(float(r.outer[:, 0].min()), 6), round(float(r.outer[:, 1].min()), 6)))
    return out


def parse_annotation_text(text: str) -> dict[str, str]:
    """Extract ``KEY=VALUE`` pairs from free text (one per line or ';'-separated)."""
    out: dict[str, str] = {}
    # MTEXT paragraph codes
    text = re.sub(r"\\[Pp]", "\n", text)
    text = re.sub(r"\\[A-Za-z][^;]*;", "", text)
    for line in re.split(r"[\n\r]+", text):
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*[=:]\s*(.+?)\s*$", line)
        if m:
            key = m.group(1).upper()
            val = m.group(2)
            out[key] = f"{out[key]};{val}" if key in out else val
    return out


# --------------------------------------------------------------------------- #
# DXF
# --------------------------------------------------------------------------- #
def _layer_ignored(layer: str, layers: list[str] | None) -> bool:
    if layers:
        return layer.upper() not in {l.upper() for l in layers}
    up = layer.upper()
    return any(p in up for p in DEFAULT_IGNORED_LAYER_PATTERNS)


def load_dxf(path: str | Path, layers: list[str] | None = None, chord_tol: float | None = None) -> Profile2D:
    import ezdxf
    from ezdxf import path as dxfpath

    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    annotations: dict[str, str] = {}
    paths = []
    for e in msp:
        kind = e.dxftype()
        if kind in ("TEXT", "MTEXT"):
            text = e.plain_text() if kind == "MTEXT" else e.dxf.text
            for k, v in parse_annotation_text(text).items():
                annotations[k] = f"{annotations[k]};{v}" if k in annotations else v
            continue
        if _layer_ignored(e.dxf.layer, layers):
            continue
        if kind in ("LINE", "ARC", "CIRCLE", "ELLIPSE", "LWPOLYLINE", "POLYLINE", "SPLINE"):
            try:
                paths.append(dxfpath.make_path(e))
            except (TypeError, ValueError):
                continue

    if not paths:
        raise ValueError(f"{path}: no profile geometry (LINE/ARC/CIRCLE/POLYLINE/SPLINE) found")

    # Chord tolerance relative to the drawing size.
    if chord_tol is None:
        ext = []
        for p in paths:
            ext.extend([(v.x, v.y) for v in p.control_vertices()] or [(p.start.x, p.start.y)])
        ext = np.asarray(ext)
        size = float(np.ptp(ext, axis=0).max()) or 1.0
        chord_tol = 2e-4 * size
    else:
        size = chord_tol / 2e-4

    closed, opened = [], []
    for p in paths:
        pts = np.array([(v.x, v.y) for v in p.flattening(distance=chord_tol)], dtype=float)
        if len(pts) < 2:
            continue
        if p.is_closed or np.linalg.norm(pts[0] - pts[-1]) <= 1e-9 * size:
            closed.append(pts)
        else:
            opened.append(pts)

    join_tol = max(1e-6 * size, 1e-9)
    loops, leftovers = chain_segments(opened, join_tol)
    if leftovers:
        # Retry with a looser tolerance for sloppy drafting.
        more, leftovers = chain_segments(leftovers, 1e-3 * size)
        loops += more
    regions = build_regions(closed + loops, tol=1e-9 * size)
    prof = Profile2D(regions=regions, annotations=annotations, source=str(path))
    if leftovers:
        prof.annotations.setdefault("_WARNING", f"{len(leftovers)} open chain(s) ignored")
    return prof


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #
def _shape_to_loop(shape, n_circle: int = 128) -> np.ndarray:
    if isinstance(shape, dict):
        t = shape.get("type", "polygon").lower()
        if t == "circle":
            cx, cy = shape["center"]
            r = shape["r"] if "r" in shape else shape["d"] / 2
            th = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
            return np.c_[cx + r * np.cos(th), cy + r * np.sin(th)]
        if t == "rect":
            x0, y0 = shape.get("origin", [0, 0])
            w, h = shape["width"], shape["height"]
            return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], float)
        if t == "polygon":
            return np.asarray(shape["points"], float)
        raise ValueError(f"unknown shape type '{t}'")
    return np.asarray(shape, float)


def load_json(path: str | Path) -> Profile2D:
    """JSON drawing format::

        {
          "loops": [ [[0,0],[100,0],[100,40],[0,40]],
                     {"type": "circle", "center": [20,20], "r": 5} ],
          "annotations": {"THICKNESS": "10", "MATERIAL": "AL6061"}
        }
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    loops = [_shape_to_loop(s) for s in data["loops"]]
    ann = {str(k).upper(): str(v) for k, v in data.get("annotations", {}).items()}
    return Profile2D(regions=build_regions(loops), annotations=ann, source=str(path))


def load_drawing(path: str | Path, layers: list[str] | None = None) -> Profile2D:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dxf":
        return load_dxf(path, layers=layers)
    if suffix == ".json":
        return load_json(path)
    raise ValueError(f"unsupported drawing format '{suffix}' (use .dxf or .json)")
