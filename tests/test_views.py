import math

import numpy as np
import pytest

from drawing2fea import fem
from drawing2fea.bc import FaceSelector
from drawing2fea.examples import make_angle_bracket_3view_dxf
from drawing2fea.materials import get_material
from drawing2fea.pipeline import run_pipeline
from drawing2fea.views import build_view_mesh, choose_axis, hole_specs, load_three_views

from viewdraw import rect, write_views

STEEL = get_material("STEEL")


def angle_bracket_views():
    """Base 120x60x15 with two Ø12 holes along z, wall y 45..60 up to z 70 with a Ø20 hole along y."""
    front = {"vis": [rect(0, 0, 120, 70), [(0, 15), (120, 15)], ("circle", 60, 45, 10)],
             "hid": [[(x, 0), (x, 15)] for x in (14, 26, 94, 106)]}
    top = {"vis": [rect(0, 0, 120, 60), [(0, 45), (120, 45)], ("circle", 20, 22, 6), ("circle", 100, 22, 6)],
           "hid": [[(50, 45), (50, 60)], [(70, 45), (70, 60)]]}
    side = {"vis": [[(0, 0), (60, 0), (60, 70), (45, 70), (45, 15), (0, 15), (0, 0)]],
            "hid": [[(16, 0), (16, 15)], [(28, 0), (28, 15)], [(45, 35), (60, 35)], [(45, 55), (60, 55)]]}
    exact = 120 * 60 * 15 + 120 * 15 * 55 - 2 * math.pi * 36 * 15 - math.pi * 100 * 15
    return (120, 60, 70), front, top, side, exact


@pytest.mark.parametrize("projection,side_pos", [("third", "right"), ("third", "left"),
                                                 ("first", "right"), ("first", "left")])
def test_projection_conventions_give_same_part(tmp_path, projection, side_pos):
    size, front, top, side, exact = angle_bracket_views()
    p = write_views(tmp_path / "v.dxf", size, front, top, side, projection, side_pos)
    tv = load_three_views(p)
    assert tv.projection == projection
    np.testing.assert_allclose(tv.size, size)
    assert [f.kind for f in tv.features] == ["cut", "cut", "cut"]
    assert not tv.warnings
    mesh, info = build_view_mesh(tv, 5.0)
    assert mesh.volume == pytest.approx(info["expected_volume_mm3"], rel=1e-9)
    assert mesh.volume == pytest.approx(exact, rel=5e-3)
    # the wall is at the back (y = 45..60) in every convention
    mp = fem.mass_properties(mesh, STEEL)
    y_centroid = (120 * 60 * 15 * 30 + 120 * 15 * 55 * 52.5) / (120 * 60 * 15 + 120 * 15 * 55)
    assert mp["centroid_mm"][1] == pytest.approx(y_centroid, abs=0.5)


def test_hole_ranges_from_hidden_lines(tmp_path):
    size, front, top, side, _ = angle_bracket_views()
    tv = load_three_views(write_views(tmp_path / "v.dxf", size, front, top, side))
    z_holes = [f for f in tv.cuts if f.axis == 2]
    y_hole = [f for f in tv.cuts if f.axis == 1][0]
    assert all(f.range == pytest.approx((0, 15)) for f in z_holes)
    assert y_hole.range == pytest.approx((45, 60))
    assert all(f.through for f in tv.cuts)
    assert y_hole.polygon.area == pytest.approx(math.pi * 100, rel=2e-3)


def test_boss_trims_visual_hull(tmp_path):
    # plate 80x60x10 with a Ø20 x 20 cylindrical boss on top
    front = {"vis": [[(0, 0), (80, 0), (80, 10), (50, 10), (50, 30), (30, 30), (30, 10), (0, 10), (0, 0)]]}
    side = {"vis": [[(0, 0), (60, 0), (60, 10), (40, 10), (40, 30), (20, 30), (20, 10), (0, 10), (0, 0)]]}
    top = {"vis": [rect(0, 0, 80, 60), ("circle", 40, 30, 10)]}
    tv = load_three_views(write_views(tmp_path / "b.dxf", (80, 60, 30), front, top, side))
    assert [f.kind for f in tv.features] == ["boss"]
    assert tv.features[0].range == pytest.approx((10, 30))
    mesh, _ = build_view_mesh(tv, 4.0)
    assert mesh.volume == pytest.approx(80 * 60 * 10 + math.pi * 100 * 20, rel=2e-3)


def test_counterbore_two_depths(tmp_path):
    front = {"vis": [rect(0, 0, 60, 20)],
             "hid": [[(25, 0), (25, 14)], [(35, 0), (35, 14)], [(21, 14), (21, 20)], [(39, 14), (39, 20)],
                     [(21, 14), (39, 14)]]}
    side = {"vis": [rect(0, 0, 60, 20)],
            "hid": [[(25, 0), (25, 14)], [(35, 0), (35, 14)], [(21, 14), (21, 20)], [(39, 14), (39, 20)],
                    [(21, 14), (39, 14)]]}
    top = {"vis": [rect(0, 0, 60, 60), ("circle", 30, 30, 9), ("circle", 30, 30, 5)]}
    tv = load_three_views(write_views(tmp_path / "c.dxf", (60, 60, 20), front, top, side))
    ranges = sorted(f.range for f in tv.cuts)
    assert ranges == [pytest.approx((0, 14)), pytest.approx((14, 20))]
    mesh, _ = build_view_mesh(tv, 3.0)
    exact = 60 * 60 * 20 - math.pi * 25 * 14 - math.pi * 81 * 6
    assert mesh.volume == pytest.approx(exact, rel=2e-3)


def test_blind_hole_from_below_hidden_circle(tmp_path):
    front = {"vis": [rect(0, 0, 60, 20)], "hid": [[(24, 0), (24, 12)], [(36, 0), (36, 12)], [(24, 12), (36, 12)]]}
    side = {"vis": [rect(0, 0, 40, 20)], "hid": [[(14, 0), (14, 12)], [(26, 0), (26, 12)], [(14, 12), (26, 12)]]}
    top = {"vis": [rect(0, 0, 60, 40)], "hid": [("circle", 30, 20, 6)]}
    tv = load_three_views(write_views(tmp_path / "h.dxf", (60, 40, 20), front, top, side))
    assert len(tv.cuts) == 1 and tv.cuts[0].range == pytest.approx((0, 12)) and not tv.cuts[0].through
    mesh, _ = build_view_mesh(tv, 3.0)
    assert mesh.volume == pytest.approx(60 * 40 * 20 - math.pi * 36 * 12, rel=2e-3)


def test_wedge_uses_exact_slab_axis(tmp_path):
    front = {"vis": [[(0, 0), (100, 0), (0, 50), (0, 0)]]}
    top = {"vis": [rect(0, 0, 100, 40)]}
    side = {"vis": [rect(0, 0, 40, 50)]}
    tv = load_three_views(write_views(tmp_path / "w.dxf", (100, 40, 50), front, top, side))
    assert choose_axis(tv) == 1  # slice along y: the slanted face is then exact
    mesh, info = build_view_mesh(tv, 5.0)
    assert not info["stepped_regions"]
    assert mesh.volume == pytest.approx(0.5 * 100 * 50 * 40, rel=1e-9)


def test_frame_and_role_layers(tmp_path):
    size, front, top, side, exact = angle_bracket_views()
    for kw in ({"frame": True}, {"role_layers": True}):
        tv = load_three_views(write_views(tmp_path / "f.dxf", size, front, top, side, **kw))
        mesh, _ = build_view_mesh(tv, 6.0)
        assert mesh.volume == pytest.approx(exact, rel=5e-3)


def test_inconsistent_loop_without_evidence_is_ignored(tmp_path):
    # a circle in the top view with no matching lines elsewhere (e.g. a marking)
    front = {"vis": [rect(0, 0, 50, 10)]}
    top = {"vis": [rect(0, 0, 50, 30), ("circle", 25, 15, 4)]}
    side = {"vis": [rect(0, 0, 30, 10)]}
    path = write_views(tmp_path / "i.dxf", (50, 30, 10), front, top, side)
    tv = load_three_views(path)
    assert [f.kind for f in tv.features] == ["ignored"] and tv.warnings
    assert build_view_mesh(tv, 3.0)[0].volume == pytest.approx(50 * 30 * 10)
    tv = load_three_views(path, assume_through=True)
    assert build_view_mesh(tv, 3.0)[0].volume == pytest.approx(50 * 30 * 10 - math.pi * 16 * 10, rel=3e-3)


def test_mesh_is_conforming_closed_surface(tmp_path):
    size, front, top, side, _ = angle_bracket_views()
    tv = load_three_views(write_views(tmp_path / "v.dxf", size, front, top, side))
    mesh, _ = build_view_mesh(tv, 6.0)
    assert (mesh.element_volumes() > 0).all()
    faces, _ = mesh.boundary_faces()
    e = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    assert (counts == 2).all()
    # hole selectors reach walls of holes in both directions
    sel = FaceSelector(mesh, None, 6.0, hole_specs(tv, 2, 6.0))
    for k in range(3):
        assert len(sel.select(f"hole:{k}")) > 0


def test_pipeline_three_view_example(tmp_path):
    path = make_angle_bracket_3view_dxf(tmp_path / "ab.dxf")
    res = run_pipeline(path, tmp_path / "out", overrides={"analysis": {"modal": 0},
                                                          "mesh": {"size": 8, "order": 1},
                                                          "output": {"vtk": False}}, verbose=False)
    assert res["model"]["operation"] == "views"
    assert res["drawing"]["mode"] == "three_views" and res["drawing"]["holes"] == 3
    assert res["static"]["reaction_force_N"][2] == pytest.approx(5000, rel=1e-6)
    assert (tmp_path / "out" / "report.html").exists()
