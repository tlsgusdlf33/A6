"""Command line interface.

    python -m drawing2fea run drawing.dxf [-c job.json] [-o results/] [--thickness 10] ...
    python -m drawing2fea run three_views.dxf          (front/top/side drawing -> 3D, auto-detected)
    python -m drawing2fea examples [dir]
    python -m drawing2fea materials
"""

from __future__ import annotations

import argparse
import sys

from .materials import MATERIALS


def _overrides(args) -> dict:
    o: dict = {}
    if args.thickness is not None:
        o.setdefault("model", {}).update(operation="extrude", thickness=args.thickness)
    if args.revolve is not None:
        o.setdefault("model", {}).update(operation="revolve", angle=args.revolve)
    if args.axis_x is not None:
        o.setdefault("model", {})["axis_x"] = args.axis_x
    if args.material:
        o["material"] = args.material
    if args.mesh_size is not None:
        o.setdefault("mesh", {})["size"] = args.mesh_size
    if args.order is not None:
        o.setdefault("mesh", {})["order"] = args.order
    if args.max_elements is not None:
        o.setdefault("mesh", {})["max_elements"] = args.max_elements
    if args.fix:
        o["boundary_conditions"] = [{"type": "fixed", "on": s} for s in args.fix]
    loads = []
    for spec in args.force or []:
        sel, vec = spec.rsplit(":", 1)
        loads.append({"type": "force", "on": sel, "value": [float(v) for v in vec.split(",")]})
    for spec in args.pressure or []:
        sel, val = spec.rsplit(":", 1)
        loads.append({"type": "pressure", "on": sel, "value": float(val)})
    if args.gravity:
        loads.append({"type": "gravity", "value": [float(v) for v in args.gravity.split(",")]})
    if loads:
        o["loads"] = loads
    if args.modes is not None:
        o.setdefault("analysis", {})["modal"] = args.modes
    if args.no_static:
        o.setdefault("analysis", {})["static"] = False
    if args.solver:
        o["solver"] = args.solver
    if args.views:
        o.setdefault("model", {})["operation"] = "views"
    for key, val in (("projection", args.projection), ("axis", args.slab_axis)):
        if val:
            o.setdefault("views", {})[key] = val
    if args.assume_through:
        o.setdefault("views", {})["assume_through"] = True
    if args.layers:
        o["drawing_layers"] = args.layers
    return o


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="drawing2fea",
                                 description="2D drawing -> 3D model -> FEA (static + modal), fully automatic")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run the full pipeline on a drawing (.dxf / .json)")
    r.add_argument("drawing")
    r.add_argument("-c", "--config", help="job file (.json / .yaml)")
    r.add_argument("-o", "--out", default="results", help="output directory (default: results)")
    r.add_argument("--thickness", type=float, help="extrude depth [mm]")
    r.add_argument("--revolve", type=float, metavar="DEG", help="revolve angle [deg] about x = axis_x")
    r.add_argument("--axis-x", type=float)
    r.add_argument("--views", action="store_true",
                   help="drawing is a front/top/side three-view drawing (auto-detected by default)")
    r.add_argument("--projection", choices=["third", "first"],
                   help="three-view projection convention (default: from layout, top above front = third)")
    r.add_argument("--slab-axis", choices=["x", "y", "z"], help="three-view meshing axis (default: automatic)")
    r.add_argument("--assume-through", action="store_true",
                   help="three-view: treat inner loops without hidden-line evidence as through holes")
    r.add_argument("--material", help=f"one of {', '.join(MATERIALS)}")
    r.add_argument("--mesh-size", type=float, help="target element size [mm]")
    r.add_argument("--order", type=int, choices=[1, 2], help="element order (2 = tet10, default)")
    r.add_argument("--max-elements", type=int)
    r.add_argument("--fix", action="append", metavar="SELECTOR", help="fully fixed faces (repeatable)")
    r.add_argument("--force", action="append", metavar="SEL:FX,FY,FZ", help="total force [N] (repeatable)")
    r.add_argument("--pressure", action="append", metavar="SEL:P", help="pressure [MPa] (repeatable)")
    r.add_argument("--gravity", metavar="AX,AY,AZ", help="acceleration [mm/s^2]")
    r.add_argument("--modes", type=int, help="number of natural frequencies (0 = skip)")
    r.add_argument("--no-static", action="store_true", help="skip static analysis")
    r.add_argument("--solver", choices=["auto", "direct", "amg"])
    r.add_argument("--layers", nargs="+", help="only read profile geometry from these DXF layers")
    r.add_argument("-q", "--quiet", action="store_true")

    e = sub.add_parser("examples", help="write example drawings")
    e.add_argument("dir", nargs="?", default="examples")
    sub.add_parser("materials", help="list the material library")

    args = ap.parse_args(argv)
    if args.cmd == "materials":
        print(f"{'name':10s} {'E [MPa]':>10s} {'nu':>6s} {'rho [t/mm3]':>12s} {'Sy [MPa]':>9s}")
        for m in MATERIALS.values():
            print(f"{m.name:10s} {m.E:10.0f} {m.nu:6.3f} {m.rho:12.3e} {m.yield_strength:9.0f}")
        return 0
    if args.cmd == "examples":
        from .examples import make_all

        for p in make_all(args.dir):
            print(p)
        return 0

    from .pipeline import run_pipeline

    try:
        run_pipeline(args.drawing, args.out, args.config, _overrides(args), verbose=not args.quiet)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
