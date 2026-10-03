"""Writers for STL (3D model) and legacy VTK (mesh + results, opens in ParaView)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from .model3d import Mesh3D


def write_stl(mesh: Mesh3D, path: str | Path, name: str = "drawing2fea") -> None:
    """Binary STL of the outer surface (outward normals)."""
    faces, _ = mesh.boundary_faces()
    p = mesh.nodes[faces]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    rec = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = n
    rec["v"] = p
    with open(path, "wb") as fh:
        fh.write(name.encode("ascii", "replace")[:80].ljust(80, b"\0"))
        fh.write(struct.pack("<I", len(faces)))
        fh.write(rec.tobytes())


def read_stl_triangle_count(path: str | Path) -> int:
    with open(path, "rb") as fh:
        fh.seek(80)
        return struct.unpack("<I", fh.read(4))[0]


def write_vtk(mesh: Mesh3D, path: str | Path, point_data: dict[str, np.ndarray] | None = None) -> None:
    """Legacy ASCII VTK unstructured grid. Vectors (n,3) and scalars (n,) supported."""
    nn = len(mesh.nodes)
    ne, npe = mesh.elements.shape
    cell_type = 24 if npe == 10 else 10
    lines = [
        "# vtk DataFile Version 3.0",
        "drawing2fea results",
        "ASCII",
        "DATASET UNSTRUCTURED_GRID",
        f"POINTS {nn} double",
    ]
    out = ["\n".join(lines), "\n"]
    out.append("\n".join(" ".join(f"{v:.9g}" for v in row) for row in mesh.nodes))
    out.append(f"\nCELLS {ne} {ne * (npe + 1)}\n")
    out.append("\n".join(f"{npe} " + " ".join(map(str, row)) for row in mesh.elements))
    out.append(f"\nCELL_TYPES {ne}\n")
    out.append("\n".join([str(cell_type)] * ne))
    if point_data:
        out.append(f"\nPOINT_DATA {nn}\n")
        for name, arr in point_data.items():
            arr = np.asarray(arr)
            safe = name.replace(" ", "_")
            if arr.ndim == 2 and arr.shape[1] == 3:
                out.append(f"VECTORS {safe} double\n")
                out.append("\n".join(" ".join(f"{v:.9g}" for v in row) for row in arr))
            elif arr.ndim == 2:
                out.append(f"FIELD {safe}_field 1\n{safe} {arr.shape[1]} {nn} double\n")
                out.append("\n".join(" ".join(f"{v:.9g}" for v in row) for row in arr))
            else:
                out.append(f"SCALARS {safe} double 1\nLOOKUP_TABLE default\n")
                out.append("\n".join(f"{v:.9g}" for v in arr))
            out.append("\n")
    Path(path).write_text("".join(out), encoding="ascii")
