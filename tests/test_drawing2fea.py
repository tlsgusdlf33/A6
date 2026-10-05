import json
import math

import ezdxf
import numpy as np
import pytest

from drawing2fea import fem
from drawing2fea.bc import FaceSelector, build_constraints, build_load_vector
from drawing2fea.config import config_from_annotations, resolve_config
from drawing2fea.drawing import Profile2D, build_regions, load_drawing, parse_annotation_text
from drawing2fea.examples import make_all
from drawing2fea.export import read_stl_triangle_count
from drawing2fea.materials import get_material
from drawing2fea.mesh2d import triangulate
from drawing2fea.model3d import extrude, revolve, to_quadratic
from drawing2fea.pipeline import run_pipeline

STEEL = get_material("STEEL")


def rect_profile(w, h):
    return Profile2D(build_regions([np.array([[0, 0], [w, 0], [w, h], [0, h]], float)]))


def beam_mesh(order=2, L=100.0, b=10.0, t=10.0, h=2.5):
    m = extrude(triangulate(rect_profile(L, b), h), t, h)
    return to_quadratic(m) if order == 2 else m


# --------------------------------------------------------------------------- #
# drawing input
# --------------------------------------------------------------------------- #
def test_dxf_loose_lines_and_circle(tmp_path):
    doc = ezdxf.new()
    msp = doc.modelspace()
    pts = [(0, 0), (80, 0), (80, 40), (0, 40)]
    order = [1, 3, 0, 2]  # shuffled and partly reversed segments must still chain
    for i in order:
        a, b = pts[i], pts[(i + 1) % 4]
        msp.add_line(b, a) if i % 2 else msp.add_line(a, b)
    msp.add_circle((40, 20), 10)
    msp.add_line((-5, 20), (85, 20), dxfattribs={"layer": "CENTER"})  # ignored layer
    msp.add_text("THICKNESS=5")
    path = tmp_path / "plate.dxf"
    doc.saveas(path)

    prof = load_drawing(path)
    assert len(prof.regions) == 1 and len(prof.holes) == 1
    assert prof.area == pytest.approx(80 * 40 - math.pi * 100, rel=1e-3)
    assert prof.annotations["THICKNESS"] == "5"


def test_nested_regions_island_inside_hole():
    outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], float)
    hole = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], float)
    island = np.array([[40, 40], [60, 40], [60, 60], [40, 60]], float)
    regions = build_regions([island, outer, hole])
    assert len(regions) == 2
    assert sum(r.area for r in regions) == pytest.approx(100**2 - 60**2 + 20**2)


def test_annotation_parsing_and_config():
    ann = parse_annotation_text("THK=8\\PMATERIAL=al6061\nFIX = xmin\nFORCE=hole:2:0,-100,0")
    cfg = config_from_annotations(ann)
    assert cfg["model"] == {"operation": "extrude", "thickness": 8.0}
    assert cfg["material"] == "al6061"
    assert cfg["boundary_conditions"] == [{"type": "fixed", "on": "xmin"}]
    assert cfg["loads"][0] == {"type": "force", "on": "hole:2", "value": [0.0, -100.0, 0.0]}
    full = resolve_config(ann, {"mesh": {"size": 3}}, {"material": "STEEL"})
    assert full["mesh"]["size"] == 3 and full["mesh"]["order"] == 2 and full["material"] == "STEEL"


# --------------------------------------------------------------------------- #
# modeling
# --------------------------------------------------------------------------- #
def test_triangulation_and_extrusion_volume():
    outer = np.array([[0, 0], [60, 0], [60, 30], [30, 30], [30, 50], [0, 50]], float)  # L-shape
    prof = Profile2D(build_regions([outer]))
    m2 = triangulate(prof, 3.0)
    assert m2.area == pytest.approx(prof.area, rel=1e-9)
    m3 = extrude(m2, 7.0, 3.0)
    assert (m3.element_volumes() > 0).all()
    assert m3.volume == pytest.approx(prof.area * 7.0, rel=1e-9)
    # closed surface: every boundary edge is shared by exactly two boundary faces
    faces, _ = m3.boundary_faces()
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert (counts == 2).all()


@pytest.mark.parametrize("angle", [360.0, 90.0])
def test_revolve_volume(angle):
    # solid cylinder r=20, h=30 drawn as a rectangle touching the axis
    prof = rect_profile(20, 30)
    m = revolve(triangulate(prof, 2.5), angle, 2.5)
    assert (m.element_volumes() > 0).all()
    exact = math.pi * 20**2 * 30 * angle / 360
    assert m.volume == pytest.approx(exact, rel=1e-2)


# --------------------------------------------------------------------------- #
# FEA verification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [1, 2])
def test_patch_test_uniform_tension(order):
    m = extrude(triangulate(rect_profile(20, 10), 3), 8, 3)
    m = to_quadratic(m) if order == 2 else m
    sel = FaceSelector(m)
    cons, _ = build_constraints(sel, [
        {"type": "symmetry", "on": "xmin", "normal": "x"},
        {"type": "symmetry", "on": "ymin", "normal": "y"},
        {"type": "symmetry", "on": "zmin", "normal": "z"},
    ])
    f, _ = build_load_vector(sel, STEEL, [{"type": "force", "on": "xmax", "value": [800, 0, 0]}])
    K = fem.assemble_stiffness(m, STEEL)
    u, R, _ = fem.solve_static(m, K, f, cons)
    s = fem.nodal_stress(m, STEEL, u)
    np.testing.assert_allclose(s[:, 0], 800 / 80, rtol=1e-8)
    np.testing.assert_allclose(s[:, 1:], 0, atol=1e-8)
    ux_end = u[0::3][np.isclose(m.nodes[:, 0], 20)]
    np.testing.assert_allclose(ux_end, 10 / STEEL.E * 20, rtol=1e-8)


def test_cantilever_deflection_stress_and_frequency():
    L, b, t, F = 100.0, 10.0, 10.0, -100.0
    m = beam_mesh(2, L, b, t)
    sel = FaceSelector(m)
    cons, _ = build_constraints(sel, [{"type": "fixed", "on": "xmin"}])
    f, _ = build_load_vector(sel, STEEL, [{"type": "force", "on": "xmax", "value": [0, F, 0]}])
    K = fem.assemble_stiffness(m, STEEL)
    u, R, _ = fem.solve_static(m, K, f, cons)
    I = t * b**3 / 12
    tip = u[1::3][np.isclose(m.nodes[:, 0], L)].mean()
    assert tip == pytest.approx(F * L**3 / (3 * STEEL.E * I), rel=0.01)
    # equilibrium
    assert R[cons.dofs[cons.dofs % 3 == 1]].sum() == pytest.approx(-F, rel=1e-8)
    # bending stress at mid-span top fibre
    s = fem.nodal_stress(m, STEEL, u)
    mid = np.isclose(m.nodes[:, 0], L / 2, atol=1.3) & np.isclose(m.nodes[:, 1], b)
    assert s[mid, 0].mean() == pytest.approx(-F * (L / 2) * (b / 2) / I, rel=0.05)

    M = fem.assemble_mass(m, STEEL)
    freqs, _, _ = fem.solve_modal(K, M, cons, 2, m.nodes)
    f1 = 1.8751**2 / (2 * math.pi) * math.sqrt(STEEL.E * I / (STEEL.rho * b * t * L**4))
    assert freqs[0] == pytest.approx(f1, rel=0.02)


def test_free_free_modal_has_rigid_body_modes():
    m = beam_mesh(1, 40, 10, 10, 5)
    K = fem.assemble_stiffness(m, STEEL)
    M = fem.assemble_mass(m, STEEL)
    freqs, _, _ = fem.solve_modal(K, M, fem.Constraints(np.array([], int), np.array([])), 8, m.nodes)
    assert (freqs[:6] < 1e-3 * freqs[6]).all()


def test_mass_properties_box():
    m = beam_mesh(1, 30, 20, 10, 5)
    mp = fem.mass_properties(m, STEEL)
    mass = 30 * 20 * 10 * STEEL.rho
    assert mp["mass_kg"] == pytest.approx(mass * 1000)
    np.testing.assert_allclose(mp["centroid_mm"], [15, 10, 5], atol=1e-9)
    Izz = mass * (30**2 + 20**2) / 12
    assert mp["inertia_t_mm2"][2][2] == pytest.approx(Izz, rel=1e-9)


def test_gravity_load_total():
    m = beam_mesh(2, 50, 10, 10, 5)
    f = fem.body_force_vector(m, STEEL, [0, -9810, 0])
    assert f[1::3].sum() == pytest.approx(-9810 * STEEL.rho * 5000)


def test_pressure_resultant_and_bad_selector():
    m = beam_mesh(2, 50, 10, 10, 5)
    sel = FaceSelector(m)
    f, log = build_load_vector(sel, STEEL, [{"type": "pressure", "on": "ymax", "value": 2.0}])
    assert f[1::3].sum() == pytest.approx(-2.0 * 50 * 10)
    with pytest.raises(ValueError):
        sel.select("x>1000")
    with pytest.raises(ValueError):
        sel.select("nonsense")


def test_underconstrained_model_is_reported():
    m = beam_mesh(1, 20, 10, 10, 5)
    sel = FaceSelector(m)
    cons, _ = build_constraints(sel, [{"type": "displacement", "on": "xmin", "x": 0.0}])
    f, _ = build_load_vector(sel, STEEL, [{"type": "force", "on": "xmax", "value": [10, 0, 0]}])
    K = fem.assemble_stiffness(m, STEEL)
    with pytest.raises(RuntimeError):
        fem.solve_static(m, K, f, cons, method="direct")


# --------------------------------------------------------------------------- #
# end-to-end
# --------------------------------------------------------------------------- #
def test_pipeline_end_to_end(tmp_path):
    drawings = make_all(tmp_path / "ex")
    cantilever = [p for p in drawings if p.name == "cantilever.json"][0]
    out = tmp_path / "out"
    res = run_pipeline(cantilever, out, overrides={"analysis": {"modal": 2}}, verbose=False)
    for name in ("results.json", "report.html", "model.stl", "results.vtk", "fig_von_mises.png"):
        assert (out / name).exists(), name
    assert read_stl_triangle_count(out / "model.stl") > 0
    saved = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert saved["static"]["max_displacement_mm"] == pytest.approx(res["static"]["max_displacement_mm"])
    # 200 x 20 x 10 steel cantilever, 500 N tip load: EB deflection 0.952 mm (+ shear)
    assert res["static"]["max_displacement_mm"] == pytest.approx(0.952, rel=0.03)
    assert res["static"]["reaction_force_N"][1] == pytest.approx(500, rel=1e-6)
    assert res["mass_properties"]["mass_kg"] == pytest.approx(200 * 20 * 10 * 7.85e-6)


def test_pipeline_bracket_dxf_with_holes(tmp_path):
    bracket = make_all(tmp_path)[0]
    res = run_pipeline(bracket, tmp_path / "out", overrides={"analysis": {"modal": 0},
                                                              "mesh": {"size": 6, "order": 1},
                                                              "output": {"report": False, "vtk": False}},
                       verbose=False)
    assert res["drawing"]["holes"] == 4
    assert res["material"]["name"] == "AL6061"
    assert [b["on"] for b in res["boundary_conditions"]] == ["hole:0", "hole:1"]
    assert res["static"]["reaction_force_N"][1] == pytest.approx(3000, rel=1e-6)


def test_pipeline_requires_operation(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"loops": [[[0, 0], [10, 0], [10, 10], [0, 10]]]}))
    with pytest.raises(ValueError, match="THICKNESS"):
        run_pipeline(p, tmp_path / "o", verbose=False)
