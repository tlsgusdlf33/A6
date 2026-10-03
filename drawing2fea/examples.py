"""Generate example drawings (DXF / JSON) that exercise the whole pipeline."""

from __future__ import annotations

import json
from pathlib import Path


def make_bracket_dxf(path: str | Path) -> Path:
    """Mounting plate drawn with loose LINE/ARC entities (rounded corners),
    two bolt holes on the left (fixed) and a load hole on the right."""
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    doc.layers.add("PROFILE", color=7)
    doc.layers.add("CENTER", color=1, linetype="CENTER")
    doc.layers.add("NOTES", color=3)
    W, H, R = 160.0, 60.0, 8.0
    # Tapered plate: full height at the left, 40 mm at the right end.
    h2 = 40.0
    y0r = (H - h2) / 2
    a = {"layer": "PROFILE"}
    msp.add_line((R, 0), (W - 20, y0r), dxfattribs=a)
    msp.add_line((W - 20, y0r), (W, y0r), dxfattribs=a)
    msp.add_line((W, y0r), (W, y0r + h2), dxfattribs=a)
    msp.add_line((W, y0r + h2), (W - 20, y0r + h2), dxfattribs=a)
    msp.add_line((W - 20, y0r + h2), (R, H), dxfattribs=a)
    msp.add_arc((R, H - R), R, 90, 180, dxfattribs=a)
    msp.add_line((0, H - R), (0, R), dxfattribs=a)
    msp.add_arc((R, R), R, 180, 270, dxfattribs=a)
    for c in [(20, 15), (20, 45)]:
        msp.add_circle(c, 5.5, dxfattribs=a)
        msp.add_line((c[0] - 9, c[1]), (c[0] + 9, c[1]), dxfattribs={"layer": "CENTER"})
    msp.add_circle((W - 12, H / 2), 6.0, dxfattribs=a)
    # Lightening slot built from LWPOLYLINE with bulges (semicircular ends).
    msp.add_lwpolyline([(55, 22, 0, 0, 0), (95, 25, 0, 0, 1), (95, 35, 0, 0, 0), (55, 38, 0, 0, 1)],
                       format="xyseb", close=True, dxfattribs=a)
    notes = [
        "THICKNESS=10",
        "MATERIAL=AL6061",
        "MESH=4",
        "FIX=hole:0;hole:1",
        "FORCE=hole:3:0,-3000,0",
        "MODES=4",
    ]
    for k, t in enumerate(notes):
        msp.add_text(t, height=3.0, dxfattribs={"layer": "NOTES"}).set_placement((0, -10 - 5 * k))
    path = Path(path)
    doc.saveas(path)
    return path


def make_flange_dxf(path: str | Path) -> Path:
    """Half cross-section of a flanged hub, revolved 360° about x = 0."""
    import ezdxf

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()
    pts = [(12, 0), (60, 0), (60, 12), (25, 12), (25, 50), (12, 50)]
    msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "PROFILE"})
    msp.add_line((0, -5), (0, 55), dxfattribs={"layer": "AXIS"})
    notes = "REVOLVE=360\\PMATERIAL=STEEL\\PMESH=4\\PFIX=zmin\\PFORCE=zmax:0,0,-20000\\PPRESSURE=r<=12:30\\PMODES=6"
    msp.add_mtext(notes, dxfattribs={"layer": "NOTES", "char_height": 3}).set_location((70, 40))
    path = Path(path)
    doc.saveas(path)
    return path


def make_cantilever_json(path: str | Path) -> Path:
    data = {
        "loops": [{"type": "rect", "origin": [0, 0], "width": 200, "height": 20}],
        "annotations": {
            "THICKNESS": "10",
            "MATERIAL": "STEEL",
            "MESH": "4",
            "FIX": "xmin",
            "FORCE": "xmax:0,-500,0",
            "MODES": "4",
        },
    }
    path = Path(path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def make_all(directory: str | Path) -> list[Path]:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    return [
        make_bracket_dxf(d / "bracket.dxf"),
        make_flange_dxf(d / "flange.dxf"),
        make_cantilever_json(d / "cantilever.json"),
    ]


if __name__ == "__main__":  # pragma: no cover
    for p in make_all(Path(__file__).resolve().parent.parent / "examples"):
        print(p)
