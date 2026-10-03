"""Material library (units: MPa, tonne/mm^3)."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Material:
    name: str
    E: float          # Young's modulus [MPa]
    nu: float         # Poisson's ratio [-]
    rho: float        # density [t/mm^3]
    yield_strength: float  # [MPa]

    def to_dict(self) -> dict:
        return asdict(self)


MATERIALS: dict[str, Material] = {
    m.name: m
    for m in [
        Material("STEEL", 210_000.0, 0.30, 7.85e-9, 275.0),
        Material("SS304", 193_000.0, 0.29, 8.00e-9, 215.0),
        Material("AL6061", 68_900.0, 0.33, 2.70e-9, 276.0),
        Material("TI6AL4V", 113_800.0, 0.342, 4.43e-9, 880.0),
        Material("COPPER", 117_000.0, 0.34, 8.96e-9, 70.0),
        Material("ABS", 2_200.0, 0.35, 1.04e-9, 40.0),
    ]
}

_ALIASES = {
    "S275": "STEEL", "SS400": "STEEL", "STEEL": "STEEL",
    "SUS304": "SS304", "STS304": "SS304",
    "AL": "AL6061", "ALUMINUM": "AL6061", "ALUMINIUM": "AL6061", "AL6061-T6": "AL6061",
    "TITANIUM": "TI6AL4V", "TI": "TI6AL4V",
    "CU": "COPPER",
}


def get_material(spec: str | dict | Material) -> Material:
    """Return a material from a library name or a dict of custom properties."""
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, dict):
        base = get_material(spec["name"]) if spec.get("name", "").upper() in MATERIALS else None
        fields = {
            "name": spec.get("name", "CUSTOM"),
            "E": spec.get("E", base.E if base else None),
            "nu": spec.get("nu", base.nu if base else None),
            "rho": spec.get("rho", base.rho if base else None),
            "yield_strength": spec.get("yield_strength", base.yield_strength if base else float("inf")),
        }
        missing = [k for k, v in fields.items() if v is None]
        if missing:
            raise ValueError(f"custom material is missing {missing}")
        return Material(**fields)
    key = str(spec).strip().upper()
    key = _ALIASES.get(key, key)
    if key not in MATERIALS:
        raise ValueError(f"unknown material '{spec}'. Available: {', '.join(MATERIALS)}")
    return MATERIALS[key]
