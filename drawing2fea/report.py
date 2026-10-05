"""Figures and the self-contained HTML report."""

from __future__ import annotations

import base64
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from .drawing import Profile2D  # noqa: E402
from .mesh2d import Mesh2D  # noqa: E402
from .model3d import Mesh3D  # noqa: E402

INK = "#1f2328"
MUTED = "#6e7781"
EDGE = "#8c959f"


def _style_axes3d(ax, pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 1e-9 * max(hi - lo))
    ax.set_box_aspect(span)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#d0d7de")
        axis.label.set_color(MUTED)
        axis.set_tick_params(colors=MUTED, labelsize=7)
    ax.set_xlabel("x [mm]", fontsize=8)
    ax.set_ylabel("y [mm]", fontsize=8)
    ax.set_zlabel("z [mm]", fontsize=8)
    ax.grid(False)
    ax.view_init(elev=24, azim=-58)


def _surface_plot(mesh: Mesh3D, path, title: str, values=None, disp=None, cmap="Reds", label=""):
    faces, _ = mesh.boundary_faces()
    pts = mesh.nodes + (disp if disp is not None else 0.0)
    tris = pts[faces]
    fig = plt.figure(figsize=(7.2, 5.4), dpi=130)
    ax = fig.add_subplot(projection="3d")
    if values is None:
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        light = np.array([0.4, -0.5, 0.75])
        light /= np.linalg.norm(light)
        shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
        base = np.array([0.36, 0.55, 0.78])
        colors = np.clip(shade[:, None] * base[None, :] + 0.12, 0, 1)
        coll = Poly3DCollection(tris, facecolors=colors, edgecolors=(0, 0, 0, 0.08), linewidths=0.2)
    else:
        fv = values[faces].mean(axis=1)
        norm = plt.Normalize(vmin=float(values.min()), vmax=float(values.max()) or 1.0)
        coll = Poly3DCollection(tris, facecolors=plt.get_cmap(cmap)(norm(fv)), edgecolors="none")
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.1)
        cb.set_label(label, color=INK, fontsize=9)
        cb.ax.tick_params(labelsize=8, colors=MUTED)
        cb.outline.set_visible(False)
    ax.add_collection3d(coll)
    _style_axes3d(ax, np.vstack([mesh.nodes, pts]))
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_drawing(profile: Profile2D, m2: Mesh2D, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=130)
    ax.triplot(m2.points[:, 0], m2.points[:, 1], m2.triangles, color="#c4ccd4", lw=0.4)
    for r in profile.regions:
        lp = np.vstack([r.outer, r.outer[:1]])
        ax.plot(lp[:, 0], lp[:, 1], color=INK, lw=1.6)
        for k, h in enumerate(r.holes):
            hp = np.vstack([h, h[:1]])
            ax.plot(hp[:, 0], hp[:, 1], color="#0969da", lw=1.4)
    for k, h in enumerate(profile.holes):
        c = h.mean(axis=0)
        ax.annotate(f"hole:{k}", c, ha="center", va="center", fontsize=7, color="#0969da")
    ax.set_aspect("equal")
    ax.set_title("2D profile and surface triangulation", color=INK, fontsize=11, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("x [mm]", color=MUTED, fontsize=8)
    ax.set_ylabel("y [mm]", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_views(tv, path):
    """The three views in third-angle arrangement, in part coordinates, with recognised features."""
    from .views import AXIS_NAMES, VIEW_AXES

    W, D, H = tv.size
    fig = plt.figure(figsize=(8.0, 6.4), dpi=130)
    gs = fig.add_gridspec(2, 2, width_ratios=[W, D], height_ratios=[D, H], wspace=0.25, hspace=0.25)
    axes = {"top": fig.add_subplot(gs[0, 0]), "front": fig.add_subplot(gs[1, 0]),
            "side": fig.add_subplot(gs[1, 1])}
    cuts = tv.cuts
    for role, ax in axes.items():
        v = tv.views[role]
        for poly in getattr(v.silhouette, "geoms", [v.silhouette]):
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color="#eef1f4", zorder=0)
        for pl in v.visible:
            ax.plot(pl[:, 0], pl[:, 1], color=INK, lw=1.1)
        for pl in v.hidden:
            ax.plot(pl[:, 0], pl[:, 1], color=MUTED, lw=0.9, ls=(0, (4, 2)))
        for f in tv.features:
            if f.view != role:
                continue
            c = f.polygon.centroid
            if f.kind == "cut":
                label, color = f"hole:{cuts.index(f)}", "#0969da"
            elif f.kind == "boss":
                label, color = "boss", "#1a7f37"
            else:
                label, color = "ignored", "#cf222e"
            xs, ys = f.polygon.exterior.xy
            ax.plot(xs, ys, color=color, lw=1.6)
            ax.annotate(label, (c.x, c.y), ha="center", va="center", fontsize=7, color=color,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
        a, b = VIEW_AXES[role]
        ax.set_aspect("equal")
        ax.set_title(f"{role.upper()} view ({AXIS_NAMES[a]}-{AXIS_NAMES[b]})", color=INK, fontsize=9, loc="left")
        ax.tick_params(colors=MUTED, labelsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle(f"Three-view drawing ({tv.projection}-angle projection) → part coordinates",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def make_figures(out: Path, profile, m2, mesh, results, fields) -> dict[str, str]:
    figs = {}
    figs["drawing"] = out / "fig_drawing.png"
    if m2 is None:  # three-view drawing
        plot_views(profile, figs["drawing"])
    else:
        plot_drawing(profile, m2, figs["drawing"])
    figs["model"] = out / "fig_model.png"
    _surface_plot(mesh, figs["model"], "3D model")
    size = float(np.ptp(mesh.nodes, axis=0).max())
    if "u" in fields:
        u = fields["u"]
        umag = np.linalg.norm(u, axis=1)
        scale = 0.08 * size / umag.max() if umag.max() > 0 else 0.0
        figs["von_mises"] = out / "fig_von_mises.png"
        _surface_plot(mesh, figs["von_mises"], f"Von Mises stress (deformation ×{scale:.3g})",
                      fields["von_mises"], u * scale, "Reds", "σ_vm [MPa]")
        figs["displacement"] = out / "fig_displacement.png"
        _surface_plot(mesh, figs["displacement"], f"Displacement magnitude (deformation ×{scale:.3g})",
                      umag, u * scale, "Blues", "|u| [mm]")
    for k, (fr, shape) in enumerate(zip(fields.get("freqs", []), fields.get("modes", []))):
        if k >= 4:
            break
        mag = np.linalg.norm(shape, axis=1)
        sc = 0.1 * size / mag.max() if mag.max() > 0 else 0
        figs[f"mode{k + 1}"] = out / f"fig_mode{k + 1}.png"
        _surface_plot(mesh, figs[f"mode{k + 1}"], f"Mode {k + 1}: {fr:.4g} Hz", mag, shape * sc, "Purples",
                      "relative amplitude")
    return {k: str(v) for k, v in figs.items()}


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _img(path: str) -> str:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(Path(path).stem)}">'


def _table(rows: list[tuple]) -> str:
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table>{body}</table>"


def _fmt(v, digits=4):
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(_fmt(x, digits) for x in v) + ")"
    return str(v)


def write_html_report(path: Path, results: dict, figs: dict[str, str]) -> None:
    r = results
    st = r.get("static")
    md = r.get("modal")
    sections = []

    verdict = ""
    if st:
        sf = st["safety_factor"]
        ok = sf >= 1.5
        label = "PASS" if ok else ("MARGINAL" if sf >= 1.0 else "FAIL")
        cls = "ok" if ok else ("warn" if sf >= 1.0 else "bad")
        verdict = (f'<div class="tiles">'
                   f'<div class="tile"><span>Max von Mises</span><b>{st["max_von_mises_MPa"]:.4g} MPa</b></div>'
                   f'<div class="tile"><span>Max displacement</span><b>{st["max_displacement_mm"]:.4g} mm</b></div>'
                   f'<div class="tile {cls}"><span>Safety factor (yield)</span><b>{sf:.3g} · {label}</b></div>')
        if md:
            verdict += f'<div class="tile"><span>1st natural frequency</span><b>{md["frequencies_Hz"][0]:.4g} Hz</b></div>'
        verdict += "</div>"
    elif md:
        verdict = (f'<div class="tiles"><div class="tile"><span>1st natural frequency</span>'
                   f'<b>{md["frequencies_Hz"][0]:.4g} Hz</b></div></div>')

    dr = r["drawing"]
    ann = "; ".join(f"{k}={v}" for k, v in dr["annotations"].items()) or "—"
    if dr.get("mode") == "three_views":
        rows = [
            ("Source", dr["source"]),
            ("Views", f'front / top / side, {dr["projection"]}-angle projection, side view {dr["side_view"]}'),
            ("Part size (W × D × H)", " × ".join(f"{v:.4g}" for v in dr["size_mm"]) + " mm"),
        ]
        rows += [("Feature", f) for f in dr["features"]] or [("Features", "—")]
        rows.append(("Annotations", ann))
        sections.append("<h2>1. Three-view drawing → features</h2>" + _table(rows) + _img(figs["drawing"]))
    else:
        sections.append("<h2>1. Drawing → profile</h2>" + _table([
            ("Source", dr["source"]),
            ("Regions / holes", f'{dr["regions"]} / {dr["holes"]}'),
            ("Profile area", f'{dr["area_mm2"]:.6g} mm²'),
            ("Bounding box", f'{_fmt(dr["bbox_min"])} – {_fmt(dr["bbox_max"])} mm'),
            ("Annotations", ann),
        ]) + _img(figs["drawing"]))

    mdl = r["model"]
    op = mdl["operation"]
    if op == "extrude":
        op_desc = f'extrude, thickness {mdl["thickness"]:g} mm'
    elif op == "views":
        op_desc = (f'reconstructed from 3 views: outline prisms intersected, features applied; '
                   f'{mdl["slabs"]} slabs along {mdl["slab_axis"]}')
    else:
        op_desc = f'revolve {mdl["angle"]:g}° about x = {mdl["axis_x"]:g}'
    mp = r["mass_properties"]
    sections.append("<h2>2. 3D model</h2>" + _table([
        ("Operation", op_desc),
        ("Volume", f'{mp["volume_mm3"]:.6g} mm³'),
        ("Mass", f'{mp["mass_kg"]:.5g} kg'),
        ("Centroid", f'{_fmt(mp["centroid_mm"])} mm'),
        ("Mesh", f'{r["mesh"]["elements"]} × {r["mesh"]["element_type"]}, {r["mesh"]["nodes"]} nodes, '
                 f'{r["mesh"]["dofs"]} DOF, target size {r["mesh"]["size_mm"]:.3g} mm'),
    ]) + _img(figs["model"]))

    mat = r["material"]
    bc_rows = [("Material", f'{mat["name"]}: E = {mat["E"]:g} MPa, ν = {mat["nu"]:g}, '
                            f'ρ = {mat["rho"]:.3g} t/mm³, σy = {mat["yield_strength"]:g} MPa')]
    for b in r["boundary_conditions"]:
        bc_rows.append(("Constraint", f'{b["type"]} [{b["components"]}] on "{b["on"]}" — {b["faces"]} faces'))
    for ld in r["loads"]:
        if ld["type"] == "force":
            bc_rows.append(("Load", f'force {_fmt(ld["total_N"])} N on "{ld["on"]}" ({ld["area_mm2"]:.4g} mm²)'))
        elif ld["type"] == "pressure":
            bc_rows.append(("Load", f'pressure {ld["pressure_MPa"]:g} MPa on "{ld["on"]}"'))
        else:
            bc_rows.append(("Load", f'gravity {_fmt(ld["accel_mm_s2"])} mm/s²'))
    sections.append("<h2>3. Material, constraints & loads</h2>" + _table(bc_rows))

    if st:
        sections.append("<h2>4. Static analysis</h2>" + _table([
            ("Max von Mises", f'{st["max_von_mises_MPa"]:.5g} MPa at {_fmt(st["max_von_mises_at"])} mm'),
            ("Max principal (σ1 / σ3)", f'{st["max_principal_MPa"]:.5g} / {st["min_principal_MPa"]:.5g} MPa'),
            ("Max displacement", f'{st["max_displacement_mm"]:.5g} mm at {_fmt(st["max_displacement_at"])} mm'),
            ("Safety factor vs yield", f'{st["safety_factor"]:.4g}'),
            ("Reaction force sum", f'{_fmt(st["reaction_force_N"])} N'),
            ("Applied load sum", f'{_fmt(st["applied_force_N"])} N'),
            ("Solver", st["solver"]),
        ]) + _img(figs["von_mises"]) + _img(figs["displacement"]))
    if md:
        rows = [("Mode", "Frequency [Hz]")] + [(i + 1, f"{f:.5g}") for i, f in enumerate(md["frequencies_Hz"])]
        imgs = "".join(_img(figs[k]) for k in sorted(figs) if k.startswith("mode"))
        sections.append("<h2>5. Modal analysis</h2>" + _table(rows) + f'<div class="grid">{imgs}</div>')
    if r.get("warnings"):
        sections.append("<h2>Notes & assumptions</h2><ul>" +
                        "".join(f"<li>{html.escape(w)}</li>" for w in r["warnings"]) + "</ul>")
    timing = ", ".join(f"{k} {v:.2f}s" for k, v in r["timing_s"].items())
    sections.append(f'<p class="muted">Timing: {html.escape(timing)} · units mm, N, MPa, t, s · drawing2fea {r["version"]}</p>')

    doc = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drawing FEA Report</title>
<style>
:root {{ --bg:#ffffff; --ink:#1f2328; --muted:#6e7781; --line:#d0d7de; --tile:#f6f8fa;
        --ok:#1a7f37; --warn:#9a6700; --bad:#cf222e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d1117; --ink:#e6edf3; --muted:#8d96a0; --line:#30363d;
        --tile:#161b22; --ok:#3fb950; --warn:#d29922; --bad:#f85149; }} img {{ filter: invert(0.92) hue-rotate(180deg); }} }}
body {{ background:var(--bg); color:var(--ink); font:15px/1.5 system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif;
       max-width:980px; margin:0 auto; padding:24px 16px 48px; }}
h1 {{ font-size:24px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:32px 0 10px; border-bottom:1px solid var(--line); padding-bottom:4px; }}
table {{ border-collapse:collapse; width:100%; margin:8px 0 12px; font-variant-numeric:tabular-nums; }}
td {{ border-bottom:1px solid var(--line); padding:6px 8px; vertical-align:top; }} td:first-child {{ color:var(--muted); width:220px; }}
img {{ max-width:100%; height:auto; display:block; margin:8px auto; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:8px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:10px; margin:16px 0; }}
.tile {{ background:var(--tile); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }}
.tile span {{ display:block; color:var(--muted); font-size:12px; }} .tile b {{ font-size:18px; }}
.tile.ok b {{ color:var(--ok); }} .tile.warn b {{ color:var(--warn); }} .tile.bad b {{ color:var(--bad); }}
.muted {{ color:var(--muted); font-size:13px; }}
</style></head><body>
<h1>2D drawing → 3D model → FEA report</h1>
<p class="muted">{html.escape(r["drawing"]["source"])}</p>
{verdict}
{''.join(sections)}
</body></html>"""
    path.write_text(doc, encoding="utf-8")
