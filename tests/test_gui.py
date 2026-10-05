import json
import plistlib
import sys
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pytest

from drawing2fea.examples import make_angle_bracket_3view_dxf, make_bracket_dxf
from drawing2fea.gui.app import App, options_to_overrides, serve
from drawing2fea.gui.shortcut import create_linux, create_macos


@pytest.fixture()
def server(tmp_path):
    app = App(tmp_path / "work")
    srv = serve(app)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield app, base
    srv.shutdown()


def call(url, data=None, method=None, headers=None):
    req = Request(url, data=data, method=method or ("POST" if data is not None else "GET"), headers=headers or {})
    try:
        with urlopen(req, timeout=60) as r:
            body = r.read()
            return r.status, (json.loads(body) if r.headers.get_content_type() == "application/json" else body)
    except HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_options_mapping():
    o = options_to_overrides({
        "operation": "extrude", "thickness": "8", "material": "AL6061", "mesh_size": "3", "order": "1",
        "fixes": ["hole:0", " ", "xmin & y>5"], "modes": "3", "static": True, "projection": "first",
        "loads": [{"type": "force", "on": "hole:2", "value": "0, 0,-500"},
                  {"type": "pressure", "on": "ymax", "value": "1.5"},
                  {"type": "gravity", "value": ""},
                  {"type": "force", "on": "", "value": ""}],
    })
    assert o["model"] == {"operation": "extrude", "thickness": 8.0}
    assert o["material"] == "AL6061" and o["mesh"] == {"size": 3.0, "order": 1}
    assert o["boundary_conditions"] == [{"type": "fixed", "on": "hole:0"}, {"type": "fixed", "on": "xmin & y>5"}]
    assert o["loads"] == [{"type": "force", "on": "hole:2", "value": [0.0, 0.0, -500.0]},
                          {"type": "pressure", "on": "ymax", "value": 1.5},
                          {"type": "gravity", "value": [0.0, 0.0, -9810.0]}]
    assert o["analysis"] == {"modal": 3} and o["views"] == {"projection": "first"}
    # empty form = keep everything from the drawing
    assert options_to_overrides({"operation": "auto", "fixes": [""], "loads": []}) == {}
    with pytest.raises(ValueError):
        options_to_overrides({"loads": [{"type": "force", "on": "xmax", "value": "1,2"}]})
    with pytest.raises(ValueError):
        options_to_overrides({"operation": "extrude", "thickness": ""})


def test_upload_preview_run_and_download(server, tmp_path):
    app, base = server
    status, info = call(base + "/api/info")
    assert status == 200 and any(m["name"] == "STEEL" for m in info["materials"])

    dxf = make_angle_bracket_3view_dxf(tmp_path / "ab.dxf").read_bytes()
    status, job = call(base + "/api/upload?name=" + quote("앵글 브래킷.dxf"), data=dxf)
    assert status == 200, job
    p = job["preview"]
    assert p["ok"] and p["mode"] == "views" and p["holes"] == ["hole:0", "hole:1", "hole:2"]
    assert p["annotations"]["MATERIAL"] == "STEEL"
    status, png = call(base + p["image"])
    assert status == 200 and png[:4] == b"\x89PNG"

    opts = {"operation": "auto", "mesh_size": "10", "order": "1", "modes": "2", "material": "AL6061"}
    status, job = call(f"{base}/api/jobs/{quote(job['id'])}/run", data=json.dumps(opts).encode())
    assert status == 200 and job["status"] == "running"
    assert "앵글_브래킷" in job["id"]
    for _ in range(600):
        status, job = call(f"{base}/api/jobs/{quote(job['id'])}")
        if job["status"] != "running":
            break
        time.sleep(0.2)
    assert job["status"] == "done", job.get("error")
    assert job["step"] == 5
    assert job["results"]["material"]["name"] == "AL6061"            # form overrides drawing
    assert job["results"]["static"]["reaction_force_N"][2] == pytest.approx(5000, rel=1e-6)  # drawing FORCE kept
    assert len(job["results"]["modal"]["frequencies_Hz"]) == 2
    for name in ("report.html", "model.stl", "results.vtk", "results.json"):
        status, body = call(base + job["files"][name])
        assert status == 200 and (body.get("static") if isinstance(body, dict) else len(body) > 100)
    assert (Path(job["folder"]) / "options.json").exists()


def test_profile_drawing_and_errors(server, tmp_path):
    app, base = server
    status, job = call(base + "/api/upload?name=b.dxf", data=make_bracket_dxf(tmp_path / "b.dxf").read_bytes())
    assert status == 200 and job["preview"]["mode"] == "profile" and len(job["preview"]["holes"]) == 4
    # failing analysis is reported, not crashing the server
    bad = {"fixes": ["hole:99"], "mesh_size": "8", "order": "1", "modes": "0"}
    call(f"{base}/api/jobs/{job['id']}/run", data=json.dumps(bad).encode())
    for _ in range(300):
        _, job = call(f"{base}/api/jobs/{job['id']}")
        if job["status"] != "running":
            break
        time.sleep(0.2)
    assert job["status"] == "error" and "hole index 99" in job["error"]

    status, err = call(base + "/api/upload?name=notes.txt", data=b"hello")
    assert status == 400 and "DXF" in err["error"]
    status, _ = call(base + "/api/jobs/../../etc")
    assert status == 404
    status, _ = call(f"{base}/files/{job['id']}/..%2F..%2Fsecret")
    assert status == 404
    status, _ = call(base + "/api/info", headers={"Host": "evil.example:80"})
    assert status == 403


def test_example_endpoint(server):
    _, base = server
    status, job = call(base + "/api/example/flange.dxf", data=b"")
    assert status == 200 and job["preview"]["operation"] == "revolve"


def test_shortcut_files(tmp_path):
    (lin,) = create_linux(tmp_path / "lin")
    text = lin.read_text(encoding="utf-8")
    assert "drawing2fea.gui" in text and "icon.png" in text
    if sys.platform != "win32":
        assert lin.stat().st_mode & 0o111
    (app,) = create_macos(tmp_path / "mac")
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleExecutable"] == "Drawing2FEA"
    assert (app / "Contents" / "Resources" / "icon.icns").stat().st_size > 1000
    assert "drawing2fea.gui" in (app / "Contents" / "MacOS" / "Drawing2FEA").read_text(encoding="utf-8")
