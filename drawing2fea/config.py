"""Job configuration: defaults <- drawing annotations <- config file <- CLI overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path

DEFAULTS: dict = {
    "model": {"operation": None, "thickness": None, "angle": 360.0, "axis_x": 0.0},
    "mesh": {"size": None, "order": 2, "max_elements": 20_000},
    "material": "STEEL",
    "boundary_conditions": [],
    "loads": [],
    "analysis": {"static": True, "modal": 6},
    "solver": "auto",
    "output": {"stl": True, "vtk": True, "report": True},
    "layers": None,
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _floats(text: str) -> list[float]:
    return [float(v) for v in text.replace(" ", "").split(",") if v]


def config_from_annotations(ann: dict[str, str]) -> dict:
    """Translate drawing text annotations (KEY=VALUE) into a partial job config.

    Recognised keys::

        THICKNESS / THK = 10           extrude depth [mm]
        REVOLVE / ANGLE = 360          revolve about x = AXIS_X
        AXIS_X = 0
        MATERIAL / MAT = AL6061
        MESH / MESH_SIZE = 2.5         target element size [mm]
        ORDER = 1|2
        FIX = xmin; hole:0             fully fixed faces (';' separates several)
        FORCE = xmax: 0,-1000,0        total force [N] on faces
        PRESSURE = ymax: 2.0           pressure [MPa] (positive pushes inward)
        GRAVITY = 0,-9810,0            acceleration [mm/s^2]
        MODES = 6                      number of natural frequencies (0 = off)
    """
    cfg: dict = {}
    a = {k.upper(): v for k, v in ann.items()}

    def put(path, value):
        d = cfg
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = value

    thk = a.get("THICKNESS", a.get("THK"))
    if thk:
        put(["model", "operation"], "extrude")
        put(["model", "thickness"], float(thk.split(";")[0]))
    if "OPERATION" in a:
        put(["model", "operation"], a["OPERATION"].strip().lower())
    rev = a.get("REVOLVE", a.get("ANGLE"))
    if rev:
        put(["model", "operation"], "revolve")
        put(["model", "angle"], float(rev.split(";")[0]))
    if "AXIS_X" in a:
        put(["model", "axis_x"], float(a["AXIS_X"]))
    mat = a.get("MATERIAL", a.get("MAT"))
    if mat:
        cfg["material"] = mat.split(";")[0].strip()
    mesh = a.get("MESH_SIZE", a.get("MESH"))
    if mesh:
        put(["mesh", "size"], float(mesh.split(";")[0]))
    if "ORDER" in a:
        put(["mesh", "order"], int(a["ORDER"]))
    if "MODES" in a:
        put(["analysis", "modal"], int(a["MODES"]))

    bcs = []
    for sel in a.get("FIX", "").split(";"):
        if sel.strip():
            bcs.append({"type": "fixed", "on": sel.strip()})
    if bcs:
        cfg["boundary_conditions"] = bcs

    loads = []
    for item in a.get("FORCE", "").split(";"):
        if item.strip():
            sel, vec = item.rsplit(":", 1)
            loads.append({"type": "force", "on": sel.strip(), "value": _floats(vec)})
    for item in a.get("PRESSURE", "").split(";"):
        if item.strip():
            sel, val = item.rsplit(":", 1)
            loads.append({"type": "pressure", "on": sel.strip(), "value": float(val)})
    if a.get("GRAVITY"):
        loads.append({"type": "gravity", "value": _floats(a["GRAVITY"].split(";")[0])})
    if loads:
        cfg["loads"] = loads
    return cfg


def load_config_file(path: str | Path | None) -> dict:
    if not path:
        return {}
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML config requires PyYAML (pip install pyyaml) - or use JSON") from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def resolve_config(annotations: dict[str, str], file_cfg: dict | None = None, overrides: dict | None = None) -> dict:
    cfg = deep_merge(DEFAULTS, config_from_annotations(annotations))
    if file_cfg:
        cfg = deep_merge(cfg, file_cfg)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg
