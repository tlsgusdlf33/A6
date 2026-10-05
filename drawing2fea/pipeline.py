"""End-to-end automation: drawing -> 3D model -> mesh -> FEA -> outputs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import __version__
from . import fem
from .bc import FaceSelector, build_constraints, build_load_vector
from .config import load_config_file, resolve_config
from .drawing import Profile2D, load_drawing
from .export import write_stl, write_vtk
from .materials import get_material
from .mesh2d import triangulate
from .model3d import extrude, remove_unused_nodes, revolve, to_quadratic
from .views import (ThreeViewDrawing, build_view_mesh, hole_specs, load_three_views,
                    looks_like_three_views, read_strokes)

_EXTRUDE_REVOLVE_KEYS = ("THICKNESS", "THK", "REVOLVE", "ANGLE")


def _auto_mesh_size(profile: Profile2D, cfg: dict) -> float:
    lo, hi = profile.bounds
    size = cfg["mesh"]["size"]
    if size:
        return float(size)
    h = float(max(hi - lo)) / 30.0
    if cfg["model"]["operation"] == "extrude":
        h = min(h, cfg["model"]["thickness"] / 2.0)
    return h


def _estimate_elements(profile: Profile2D, cfg: dict, h: float) -> float:
    tris = 2.4 * profile.area / (h * h)
    if cfg["model"]["operation"] == "extrude":
        layers = max(2, np.ceil(cfg["model"]["thickness"] / h))
    else:
        lo, hi = profile.bounds
        rmax = hi[0] - cfg["model"]["axis_x"]
        layers = max(16, np.ceil(np.radians(cfg["model"]["angle"]) * rmax / h))
    return 3.0 * tris * layers


def build_model(profile: Profile2D, cfg: dict, warnings: list[str]):
    """Create the 2D triangulation and the 3D tetrahedral mesh."""
    model = cfg["model"]
    if model["operation"] is None:
        if model.get("thickness"):
            model["operation"] = "extrude"
        else:
            raise ValueError(
                "modeling operation is unknown: add THICKNESS=<mm> (extrude) or REVOLVE=<deg> "
                "text to the drawing, or set model.operation in the config file"
            )
    op = model["operation"]
    if op == "extrude" and not model.get("thickness"):
        raise ValueError("extrude needs model.thickness (or THICKNESS=... in the drawing)")
    if op not in ("extrude", "revolve"):
        raise ValueError(f"unknown modeling operation '{op}'")

    h = _auto_mesh_size(profile, cfg)
    max_el = cfg["mesh"]["max_elements"]
    est = _estimate_elements(profile, cfg, h)
    if est > max_el:
        new_h = h * (est / max_el) ** (1 / 3)
        warnings.append(f"mesh size increased from {h:.3g} to {new_h:.3g} mm to stay under "
                        f"{max_el} elements (raise mesh.max_elements for a finer mesh)")
        h = new_h

    m2 = triangulate(profile, h)
    if op == "extrude":
        mesh = extrude(m2, float(model["thickness"]), h, cfg.get("layers"))
    else:
        mesh = revolve(m2, float(model["angle"]), h, float(model["axis_x"]))
    mesh = remove_unused_nodes(mesh)
    if int(cfg["mesh"]["order"]) == 2:
        mesh = to_quadratic(mesh)
    return m2, mesh, h


def build_views_model(tv: ThreeViewDrawing, cfg: dict, warnings: list[str]):
    """3D solid + tet mesh reconstructed from a front/top/side drawing."""
    axis_cfg = cfg["views"].get("axis")
    axis = "xyz".index(axis_cfg.lower()) if axis_cfg else None
    h = float(cfg["mesh"]["size"] or tv.size.max() / 25.0)
    max_el = int(cfg["mesh"]["max_elements"])
    h0 = h
    for _ in range(6):
        mesh, info = build_view_mesh(tv, h, axis)
        if len(mesh.elements) <= max_el:
            break
        h *= (len(mesh.elements) / max_el) ** (1 / 3) * 1.05
    if h != h0:
        warnings.append(f"mesh size increased from {h0:.3g} to {h:.3g} mm to stay under "
                        f"{max_el} elements (raise mesh.max_elements for a finer mesh)")
    if abs(mesh.volume - info["expected_volume_mm3"]) > 1e-3 * info["expected_volume_mm3"]:
        warnings.append(f"mesh volume {mesh.volume:.6g} differs from the reconstructed solid "
                        f"{info['expected_volume_mm3']:.6g} mm³")
    if info["stepped_regions"]:
        warnings.append(f"curved/slanted edges seen across the slicing axis ({info['slab_axis']}) are "
                        f"approximated by steps of {info['step_size_mm']:.3g} mm")
    if int(cfg["mesh"]["order"]) == 2:
        mesh = to_quadratic(mesh)
    slab_axis = "xyz".index(info["slab_axis"])
    return mesh, h, info, hole_specs(tv, slab_axis, h)


def _wants_views(path: Path, file_cfg: dict, overrides: dict | None):
    """Decide whether a DXF is a three-view drawing. Returns (strokes, annotations) or None."""
    if path.suffix.lower() != ".dxf":
        return None
    forced = None
    for src in (file_cfg or {}, overrides or {}):
        forced = (src.get("model") or {}).get("operation", forced)
    if forced is not None and forced != "views":
        return None
    strokes, ann = read_strokes(path)
    if forced == "views" or ann.get("VIEWS") or "PROJECTION" in ann or ann.get("OPERATION", "").lower() == "views":
        return strokes, ann
    if any(k in ann for k in _EXTRUDE_REVOLVE_KEYS) or "OPERATION" in ann:
        return None
    return (strokes, ann) if looks_like_three_views(strokes) else None


def _default_bcs(cfg: dict, warnings: list[str]) -> None:
    op = cfg["model"]["operation"]
    if not cfg["boundary_conditions"] and cfg["analysis"].get("static"):
        sel = "xmin" if op == "extrude" else "zmin"  # revolve / views: bottom face
        cfg["boundary_conditions"] = [{"type": "fixed", "on": sel}]
        warnings.append(f"no constraint given: assumed fully fixed face '{sel}'")
    if not cfg["loads"] and cfg["analysis"].get("static"):
        g = [0.0, -9810.0, 0.0] if op == "extrude" else [0.0, 0.0, -9810.0]
        cfg["loads"] = [{"type": "gravity", "value": g}]
        warnings.append(f"no load given: assumed self-weight, gravity {g} mm/s²")


def run_pipeline(drawing_path: str | Path, out_dir: str | Path = "results",
                 config: str | Path | dict | None = None, overrides: dict | None = None,
                 verbose: bool = True, log=None) -> dict:
    """Run the full automation and return the results dictionary.

    ``config`` may be a path to a JSON/YAML job file or an already-loaded dict.
    ``log`` receives the progress lines (default: ``print`` when ``verbose``).
    """
    if log is None:
        log = print if verbose else (lambda *a, **k: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}
    warnings: list[str] = []

    t0 = time.perf_counter()
    file_cfg = config if isinstance(config, dict) else load_config_file(config)
    layers = (file_cfg or {}).get("drawing_layers") or (overrides or {}).get("drawing_layers")
    drawing_path = Path(drawing_path)
    views_input = _wants_views(drawing_path, file_cfg, overrides)
    profile = tv = m2 = None
    holes = None
    if views_input is not None:
        cfg = resolve_config(views_input[1], file_cfg, overrides)
        cfg["model"]["operation"] = "views"
        tv = load_three_views(drawing_path, cfg["views"].get("projection"),
                              bool(cfg["views"].get("assume_through")), views_input)
        warnings.extend(tv.warnings)
        drawing_summary = tv.summary()
        log(f"[1/5] drawing: 3 views ({tv.projection}-angle projection), part "
            f"{tv.size[0]:.4g} x {tv.size[1]:.4g} x {tv.size[2]:.4g} mm, "
            f"{len(tv.cuts)} hole(s), {sum(f.kind == 'boss' for f in tv.features)} boss(es)")
    else:
        profile = load_drawing(drawing_path, layers=layers)
        if "_WARNING" in profile.annotations:
            warnings.append(profile.annotations.pop("_WARNING"))
        cfg = resolve_config(profile.annotations, file_cfg, overrides)
        if cfg["model"]["operation"] == "views":
            raise ValueError("three-view reconstruction needs a DXF drawing with front, top and side views")
        drawing_summary = profile.summary()
        log(f"[1/5] drawing: {len(profile.regions)} region(s), {len(profile.holes)} hole(s), "
            f"area {profile.area:.6g} mm²")
    timing["read drawing"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    if tv is not None:
        mesh, h, view_info, holes = build_views_model(tv, cfg, warnings)
        cfg["model"].update(view_info)
    else:
        m2, mesh, h = build_model(profile, cfg, warnings)
    mat = get_material(cfg["material"])
    timing["3D modeling + mesh"] = time.perf_counter() - t0
    etype = "tet10" if mesh.order == 2 else "tet4"
    log(f"[2/5] 3D model: {cfg['model']['operation']}, volume {mesh.volume:.6g} mm³, "
        f"{len(mesh.elements)} {etype} / {len(mesh.nodes)} nodes")

    results: dict = {
        "version": __version__,
        "drawing": drawing_summary,
        "model": dict(cfg["model"]),
        "material": mat.to_dict(),
        "mesh": {"elements": int(len(mesh.elements)), "nodes": int(len(mesh.nodes)),
                 "dofs": int(3 * len(mesh.nodes)), "element_type": etype, "size_mm": h},
        "mass_properties": fem.mass_properties(mesh, mat),
        "config": cfg,
    }

    t0 = time.perf_counter()
    if cfg["output"].get("stl", True):
        write_stl(mesh, out / "model.stl")
    K = fem.assemble_stiffness(mesh, mat)
    timing["assembly"] = time.perf_counter() - t0

    _default_bcs(cfg, warnings)
    selector = FaceSelector(mesh, profile, h, holes)
    cons, bc_log = build_constraints(selector, cfg["boundary_conditions"])
    results["boundary_conditions"] = bc_log
    results["loads"] = []
    fields: dict = {}
    point_data: dict = {}

    if cfg["analysis"].get("static"):
        t0 = time.perf_counter()
        f, load_log = build_load_vector(selector, mat, cfg["loads"])
        results["loads"] = load_log
        u, R, desc = fem.solve_static(mesh, K, f, cons, cfg.get("solver", "auto"))
        U = u.reshape(-1, 3)
        S = fem.nodal_stress(mesh, mat, u)
        vm = fem.von_mises(S)
        P = fem.principal_stresses(S)
        umag = np.linalg.norm(U, axis=1)
        i_vm, i_u = int(np.argmax(vm)), int(np.argmax(umag))
        max_vm = float(vm[i_vm])
        results["static"] = {
            "max_von_mises_MPa": max_vm,
            "max_von_mises_at": mesh.nodes[i_vm].tolist(),
            "max_principal_MPa": float(P[:, 0].max()),
            "min_principal_MPa": float(P[:, 2].min()),
            "max_displacement_mm": float(umag[i_u]),
            "max_displacement_at": mesh.nodes[i_u].tolist(),
            "safety_factor": float(mat.yield_strength / max_vm) if max_vm > 0 else float("inf"),
            "reaction_force_N": [float(R[cons.dofs[cons.dofs % 3 == c]].sum()) for c in range(3)],
            "applied_force_N": [float(f[c::3].sum()) for c in range(3)],
            "solver": desc,
        }
        fields.update(u=U, von_mises=vm)
        point_data.update(displacement=U, von_mises=vm, stress=S, principal=P)
        timing["static solve"] = time.perf_counter() - t0
        st = results["static"]
        log(f"[3/5] static: max σ_vm {st['max_von_mises_MPa']:.4g} MPa, "
            f"max |u| {st['max_displacement_mm']:.4g} mm, SF {st['safety_factor']:.3g}")
        if st["safety_factor"] < 1.0:
            warnings.append("von Mises stress exceeds the yield strength (safety factor < 1)")
    else:
        log("[3/5] static: skipped")

    n_modes = int(cfg["analysis"].get("modal") or 0)
    if n_modes > 0:
        t0 = time.perf_counter()
        M = fem.assemble_mass(mesh, mat)
        freqs, modes, mdesc = fem.solve_modal(K, M, cons, n_modes, mesh.nodes, cfg.get("solver", "auto"))
        results["modal"] = {"frequencies_Hz": freqs.tolist(),
                            "constrained": bool(len(cons.dofs)), "solver": mdesc}
        fields["freqs"] = freqs
        fields["modes"] = [modes[:, k].reshape(-1, 3) for k in range(modes.shape[1])]
        for k in range(modes.shape[1]):
            point_data[f"mode_{k + 1}_{freqs[k]:.4g}Hz"] = fields["modes"][k]
        timing["modal solve"] = time.perf_counter() - t0
        log(f"[4/5] modal: " + ", ".join(f"{fr:.4g}" for fr in freqs) + " Hz")
    else:
        log("[4/5] modal: skipped")

    t0 = time.perf_counter()
    if cfg["output"].get("vtk", True):
        write_vtk(mesh, out / "results.vtk", point_data)
    results["warnings"] = warnings
    results["timing_s"] = timing
    if cfg["output"].get("report", True):
        from .report import make_figures, write_html_report

        figs = make_figures(out, profile if tv is None else tv, m2, mesh, results, fields)
        timing["report"] = time.perf_counter() - t0
        write_html_report(out / "report.html", results, figs)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    log(f"[5/5] outputs written to {out.resolve()}")
    for w in warnings:
        log(f"  note: {w}")
    return results
