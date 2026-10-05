"""Helpers that write synthetic three-view DXF drawings for the tests."""

import ezdxf


def write_views(path, size, front, top, side, projection="third", side_pos="right", gap=25.0,
                frame=False, role_layers=False):
    """Write a three-view DXF.

    ``front``/``top``/``side`` are dicts with ``vis``/``hid`` lists of items in part
    coordinates (front: x-z, top: x-y, side: y-z; y = 0 is the front face).
    Items are point lists (open polylines; repeat the first point to close) or
    ("circle", cx, cy, r).
    """
    W, D, H = size
    doc = ezdxf.new("R2010", setup=True)
    msp = doc.modelspace()
    doc.layers.add("HIDDEN", linetype="HIDDEN")
    for name in ("FRONT", "TOP", "SIDE", "FRONT_HIDDEN", "TOP_HIDDEN", "SIDE_HIDDEN"):
        doc.layers.add(name, linetype="HIDDEN" if name.endswith("HIDDEN") else "CONTINUOUS")

    if projection == "third":
        top_map = lambda x, y: (x, H + gap + y)
    else:
        top_map = lambda x, y: (x, -gap - D + y)
    near = projection == "third"
    if side_pos == "right":
        side_map = (lambda y, z: (W + gap + y, z)) if near else (lambda y, z: (W + gap + D - y, z))
    else:
        side_map = (lambda y, z: (-gap - y, z)) if near else (lambda y, z: (-gap - D + y, z))
    maps = {"front": lambda u, v: (u, v), "top": top_map, "side": side_map}

    for role, spec in (("front", front), ("top", top), ("side", side)):
        m = maps[role]
        for kind in ("vis", "hid"):
            if role_layers:
                layer = role.upper() + ("_HIDDEN" if kind == "hid" else "")
            else:
                layer = "HIDDEN" if kind == "hid" else "0"
            for item in spec.get(kind, []):
                if item[0] == "circle":
                    _, cx, cy, r = item
                    (px, py) = m(cx, cy)
                    # mirrored mappings keep circles circles; only the centre moves
                    msp.add_circle((px, py), r, dxfattribs={"layer": layer})
                else:
                    pts = [m(*p) for p in item]
                    msp.add_lwpolyline(pts, dxfattribs={"layer": layer})
    if frame:
        msp.add_lwpolyline([(-300, -300), (500, -300), (500, 500), (-300, 500), (-300, -300)])
    doc.saveas(path)
    return path


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
