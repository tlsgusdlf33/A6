"""drawing2fea — 2D 도면(DXF/JSON)에서 3D 모델링과 구조 해석(FEA)을 자동으로 수행.

Pipeline::

    2D drawing  ->  closed profile loops  ->  3D solid (extrude / revolve)
                ->  tetrahedral mesh (tet4 / tet10)  ->  static + modal FEA
                ->  STL / VTK / JSON / HTML report

Units are consistent mm - N - MPa - tonne(t) - s throughout.
"""

__version__ = "0.1.0"

from .drawing import Profile2D, Region, load_drawing
from .materials import Material, get_material, MATERIALS
from .pipeline import run_pipeline

__all__ = [
    "Profile2D",
    "Region",
    "load_drawing",
    "Material",
    "get_material",
    "MATERIALS",
    "run_pipeline",
]
