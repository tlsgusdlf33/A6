"""Desktop app: a local web UI with drag & drop, served on 127.0.0.1.

Double-clicking the app icon calls :func:`main`, which starts a small HTTP
server on a free local port and opens it in an app window (Edge / Chrome
``--app`` mode) or the default browser. Drawings are dropped onto the window or
chosen with the attach button; every analysis gets its own folder under
``~/Documents/Drawing2FEA/jobs``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse
from urllib.request import urlopen

from .. import __version__

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
ASSETS = HERE / "assets"
ALLOWED_SUFFIXES = {".dxf", ".json", ".dwg"}
MAX_UPLOAD = 100 * 1024 * 1024
RESULT_FILES = ("report.html", "model.stl", "results.vtk", "results.json")
_SAFE = re.compile(r"[^0-9A-Za-z가-힣._-]+")
_ID = re.compile(r"^[0-9A-Za-z가-힣._-]{1,120}$")


def default_workdir() -> Path:
    docs = Path.home() / "Documents"
    return (docs if docs.is_dir() else Path.home()) / "Drawing2FEA"


# --------------------------------------------------------------------------- #
# jobs
# --------------------------------------------------------------------------- #
class Job:
    def __init__(self, job_id: str, folder: Path, input_path: Path):
        self.id = job_id
        self.dir = folder
        self.input = input_path
        self.status = "uploaded"      # uploaded | running | done | error
        self.log: list[str] = []
        self.step = 0
        self.preview: dict = {}
        self.results: dict | None = None
        self.error: str | None = None
        self.started = self.finished = None

    @property
    def out(self) -> Path:
        return self.dir / "results"

    def to_dict(self) -> dict:
        files = {}
        for name in RESULT_FILES:
            if (self.out / name).exists():
                files[name] = f"/files/{quote(self.id)}/{name}"
        res = None
        if self.results:
            r = self.results
            res = {
                "static": r.get("static"),
                "modal": r.get("modal"),
                "mass_properties": r.get("mass_properties"),
                "mesh": r.get("mesh"),
                "material": r.get("material"),
                "warnings": r.get("warnings", []),
            }
        return {
            "id": self.id, "name": self.input.name, "status": self.status, "step": self.step,
            "log": self.log[-200:], "preview": self.preview, "results": res, "error": self.error,
            "files": files, "folder": str(self.dir),
            "elapsed": (self.finished or time.time()) - self.started if self.started else None,
        }


def options_to_overrides(opt: dict) -> dict:
    """Translate the form of the UI into a pipeline override dict (empty field = keep drawing value)."""

    def num(key, kind=float):
        v = str(opt.get(key, "") or "").strip()
        if not v:
            return None
        try:
            return kind(v)
        except ValueError:
            raise ValueError(f"'{v}' is not a valid number ({key})") from None

    def vec(text, what):
        try:
            vals = [float(v) for v in str(text).replace(" ", "").split(",") if v != ""]
        except ValueError:
            raise ValueError(f"{what}: '{text}' must be numbers separated by commas") from None
        if len(vals) != 3:
            raise ValueError(f"{what}: three components x,y,z are required (got '{text}')")
        return vals

    o: dict = {}
    op = (opt.get("operation") or "auto").lower()
    if op == "views":
        o.setdefault("model", {})["operation"] = "views"
    elif op == "extrude":
        t = num("thickness")
        if not t or t <= 0:
            raise ValueError("돌출 두께(mm)를 입력하세요")
        o["model"] = {"operation": "extrude", "thickness": t}
    elif op == "revolve":
        o["model"] = {"operation": "revolve", "angle": num("angle") or 360.0}
    proj = (opt.get("projection") or "auto").lower()
    if proj in ("third", "first"):
        o.setdefault("views", {})["projection"] = proj
    if opt.get("assume_through"):
        o.setdefault("views", {})["assume_through"] = True
    if opt.get("material"):
        o["material"] = str(opt["material"])
    if num("mesh_size") is not None:
        o.setdefault("mesh", {})["size"] = num("mesh_size")
    if str(opt.get("order", "")) in ("1", "2"):
        o.setdefault("mesh", {})["order"] = int(opt["order"])
    fixes = [f.strip() for f in opt.get("fixes", []) if str(f).strip()]
    if fixes:
        o["boundary_conditions"] = [{"type": "fixed", "on": f} for f in fixes]
    loads = []
    for ld in opt.get("loads", []):
        kind = ld.get("type", "force")
        on = str(ld.get("on", "")).strip()
        val = str(ld.get("value", "")).strip()
        if kind == "gravity":
            loads.append({"type": "gravity", "value": vec(val or "0,0,-9810", "중력")})
            continue
        if not on and not val:
            continue
        if not on:
            raise ValueError("하중 위치(선택자)를 입력하세요")
        if kind == "force":
            loads.append({"type": "force", "on": on, "value": vec(val, f"힘 ({on})")})
        elif kind == "pressure":
            try:
                loads.append({"type": "pressure", "on": on, "value": float(val)})
            except ValueError:
                raise ValueError(f"압력 ({on}): '{val}' is not a number") from None
    if loads:
        o["loads"] = loads
    modes = num("modes", int)
    if modes is not None:
        o.setdefault("analysis", {})["modal"] = modes
    if opt.get("static") is False:
        o.setdefault("analysis", {})["static"] = False
    return o


class App:
    def __init__(self, workdir: Path | None = None):
        self.workdir = Path(workdir or default_workdir())
        (self.workdir / "jobs").mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.lock = threading.Lock()
        self.run_lock = threading.Lock()
        self.last_seen = time.time()
        self.seen_client = False

    # -- uploads ------------------------------------------------------------ #
    def new_job(self, filename: str, data: bytes) -> Job:
        name = Path(filename).name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise ValueError(f"지원하지 않는 파일 형식입니다: '{suffix or name}'. DXF(.dxf) 또는 JSON 도면을 사용하세요.")
        if not data:
            raise ValueError("빈 파일입니다")
        stem = _SAFE.sub("_", Path(name).stem).strip("._") or "drawing"
        job_id = f"{datetime.now():%Y%m%d-%H%M%S}_{stem[:60]}"
        with self.lock:
            k = 2
            base = job_id
            while job_id in self.jobs or (self.workdir / "jobs" / job_id).exists():
                job_id = f"{base}-{k}"
                k += 1
            folder = self.workdir / "jobs" / job_id
            folder.mkdir(parents=True)
        path = folder / f"{stem}{suffix}"
        path.write_bytes(data)
        if suffix == ".dwg":
            path = convert_dwg(path)
        job = Job(job_id, folder, path)
        with self.lock:
            self.jobs[job_id] = job
        job.preview = make_preview(path, folder)
        return job

    def example_job(self, name: str) -> Job:
        from ..examples import make_all
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for p in make_all(tmp):
                if p.name == name:
                    return self.new_job(p.name, p.read_bytes())
        raise ValueError(f"unknown example '{name}'")

    # -- running ------------------------------------------------------------ #
    def start(self, job: Job, options: dict) -> None:
        overrides = options_to_overrides(options)
        if job.status == "running":
            raise ValueError("이미 실행 중입니다")
        job.status, job.log, job.step, job.error, job.results = "running", [], 0, None, None
        job.started, job.finished = time.time(), None
        (job.dir / "options.json").write_text(json.dumps(options, ensure_ascii=False, indent=2), encoding="utf-8")
        threading.Thread(target=self._run, args=(job, overrides), daemon=True).start()

    def _run(self, job: Job, overrides: dict) -> None:
        from ..pipeline import run_pipeline

        def log(msg, *a, **k):
            line = str(msg)
            m = re.match(r"\[(\d)/5\]", line)
            if m:
                job.step = int(m.group(1))
            job.log.append(line)

        with self.run_lock:  # one analysis at a time keeps memory use predictable
            try:
                if job.out.exists():
                    shutil.rmtree(job.out)
                job.log.append("해석을 시작합니다…")
                job.results = run_pipeline(job.input, job.out, overrides=overrides, log=log)
                job.status = "done"
            except Exception as exc:  # noqa: BLE001 - shown to the user
                job.error = str(exc) or exc.__class__.__name__
                job.log.append("오류: " + job.error)
                (job.dir / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
                job.status = "error"
            finally:
                job.finished = time.time()


def convert_dwg(path: Path) -> Path:
    """Convert DWG to DXF with the ODA File Converter, if it is installed."""
    try:
        from ezdxf.addons import odafc

        out = path.with_suffix(".dxf")
        odafc.convert(str(path), str(out), version="R2018")
        return out
    except Exception as exc:  # noqa: BLE001
        raise ValueError("DWG 파일은 ODA File Converter(무료)가 설치되어 있어야 읽을 수 있습니다. "
                         "CAD에서 'DXF로 저장' 후 다시 첨부하세요.") from exc


def make_preview(path: Path, folder: Path) -> dict:
    """Recognise the drawing (no analysis) and draw a preview with hole numbers."""
    from ..config import resolve_config
    from ..drawing import load_drawing
    from ..mesh2d import triangulate
    from ..pipeline import _wants_views
    from ..report import plot_drawing, plot_views
    from ..views import load_three_views

    img = folder / "preview.png"
    info: dict = {"image": None, "warnings": []}
    try:
        vi = _wants_views(path, {}, None)
        if vi is not None:
            tv = load_three_views(path, None, False, vi)
            plot_views(tv, img)
            info.update(mode="views", projection=tv.projection, size=[round(float(v), 4) for v in tv.size],
                        features=[(f"hole:{tv.cuts.index(f)} · " if f.kind == "cut" else "") + f.describe_ko()
                                  for f in tv.features],
                        holes=[f"hole:{k}" for k in range(len(tv.cuts))],
                        annotations=tv.annotations, warnings=list(tv.warnings))
        else:
            prof = load_drawing(path)
            lo, hi = prof.bounds
            plot_drawing(prof, triangulate(prof, float(max(hi - lo)) / 40.0), img)
            cfg = resolve_config(prof.annotations)
            info.update(mode="profile", regions=len(prof.regions), area=prof.area,
                        size=[round(float(v), 4) for v in (hi - lo)],
                        holes=[f"hole:{k}" for k in range(len(prof.holes))],
                        annotations={k: v for k, v in prof.annotations.items() if not k.startswith("_")},
                        operation=cfg["model"]["operation"])
            if cfg["model"]["operation"] is None:
                info["warnings"].append("모델링 방식을 정하세요: 돌출 두께 또는 회전 각도를 입력하거나, "
                                        "3면도라면 '3면도'를 선택하세요.")
        info["image"] = f"/files/{quote(folder.name)}/preview.png"
        info["ok"] = True
    except Exception as exc:  # noqa: BLE001
        info.update(ok=False, error=str(exc))
    return info


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
_MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".png": "image/png", ".ico": "image/x-icon",
         ".svg": "image/svg+xml", ".json": "application/json; charset=utf-8",
         ".stl": "model/stl", ".vtk": "application/octet-stream", ".dxf": "application/dxf"}


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"Drawing2FEA/{__version__}"

        def log_message(self, *args):  # keep the console quiet
            pass

        # -- helpers -------------------------------------------------------- #
        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False, default=float).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _error(self, msg: str, code: int = 400):
            self._json({"error": msg}, code)

        def _file(self, path: Path, download: bool = False):
            if not path.is_file():
                return self._error("not found", 404)
            extra = {"Content-Disposition": f'attachment; filename="{path.name}"'} if download else None
            self._send(200, path.read_bytes(), _MIME.get(path.suffix.lower(), "application/octet-stream"), extra)

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError("파일이 너무 큽니다 (최대 100 MB)")
            return self.rfile.read(n) if n else b""

        def _job(self, job_id: str) -> Job | None:
            return app.jobs.get(job_id) if _ID.match(job_id or "") else None

        def _local_only(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            if host not in ("127.0.0.1", "localhost"):
                self._error("forbidden", 403)
                return False
            return True

        # -- routes ------------------------------------------------------- #
        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            if not self._local_only():
                return
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            app.last_seen = time.time()
            if not parts:
                return self._file(STATIC / "index.html")
            if parts[0] == "static" and len(parts) == 2:
                return self._file(STATIC / Path(parts[1]).name)
            if parts[0] == "assets" and len(parts) == 2:
                return self._file(ASSETS / Path(parts[1]).name)
            if parts == ["api", "info"]:
                from ..materials import MATERIALS

                return self._json({"version": __version__, "workdir": str(app.workdir),
                                   "materials": [m.to_dict() for m in MATERIALS.values()],
                                   "examples": ["angle_bracket_3view.dxf", "bracket.dxf", "flange.dxf",
                                                "cantilever.json"]})
            if len(parts) == 3 and parts[:2] == ["api", "jobs"]:
                job = self._job(parts[2])
                return self._json(job.to_dict()) if job else self._error("unknown job", 404)
            if parts[0] == "files" and len(parts) == 3:
                job = self._job(parts[1])
                name = Path(parts[2]).name
                if not job or name != parts[2]:
                    return self._error("not found", 404)
                q = parse_qs(url.query)
                base = job.dir if name in ("preview.png",) or name == job.input.name else job.out
                return self._file(base / name, download="download" in q)
            return self._error("not found", 404)

        def do_POST(self):
            if not self._local_only():
                return
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.split("/") if p]
            app.last_seen = time.time()
            app.seen_client = True
            try:
                if parts == ["api", "heartbeat"]:
                    return self._json({"ok": True})
                if parts == ["api", "shortcut"]:
                    from .shortcut import create_shortcut

                    return self._json({"created": [str(p) for p in create_shortcut()]})
                if parts == ["api", "upload"]:
                    name = parse_qs(url.query).get("name", ["drawing.dxf"])[0]
                    job = app.new_job(name, self._body())
                    return self._json(job.to_dict())
                if parts[:2] == ["api", "example"] and len(parts) == 3:
                    return self._json(app.example_job(parts[2]).to_dict())
                if len(parts) == 4 and parts[:2] == ["api", "jobs"]:
                    job = self._job(parts[2])
                    if not job:
                        return self._error("unknown job", 404)
                    if parts[3] == "run":
                        options = json.loads(self._body() or b"{}")
                        app.start(job, options)
                        return self._json(job.to_dict())
                    if parts[3] == "open":
                        open_folder(job.out if job.out.exists() else job.dir)
                        return self._json({"ok": True})
                return self._error("not found", 404)
            except ValueError as exc:
                return self._error(str(exc))
            except Exception as exc:  # noqa: BLE001
                return self._error(f"{exc.__class__.__name__}: {exc}", 500)

    return Handler


def open_folder(path: Path) -> None:
    if sys.platform.startswith("win"):
        os.startfile(str(path))  # noqa: S606
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def serve(app: App, port: int = 0) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


# --------------------------------------------------------------------------- #
# window
# --------------------------------------------------------------------------- #
def _browser_candidates() -> list[str]:
    c: list[str] = []
    if sys.platform.startswith("win"):
        for base in (os.environ.get("PROGRAMFILES(X86)"), os.environ.get("PROGRAMFILES"),
                     os.environ.get("LOCALAPPDATA")):
            if base:
                c += [os.path.join(base, "Microsoft", "Edge", "Application", "msedge.exe"),
                      os.path.join(base, "Google", "Chrome", "Application", "chrome.exe")]
    elif sys.platform == "darwin":
        c += ["/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
              "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
              "/Applications/Chromium.app/Contents/MacOS/Chromium"]
    else:
        for name in ("microsoft-edge", "google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
            p = shutil.which(name)
            if p:
                c.append(p)
    return [p for p in c if p and os.path.exists(p)]


def open_window(url: str, workdir: Path) -> None:
    """Open the UI as a standalone app window if Edge/Chrome exist, else in the default browser."""
    for exe in _browser_candidates():
        try:
            subprocess.Popen([exe, f"--app={url}", "--window-size=1280,900", "--no-first-run",
                              "--no-default-browser-check", f"--user-data-dir={workdir / '.window'}"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except OSError:
            continue
    webbrowser.open(url)


def _existing_instance(workdir: Path) -> str | None:
    try:
        info = json.loads((workdir / ".server.json").read_text(encoding="utf-8"))
        url = f"http://127.0.0.1:{int(info['port'])}/"
        with urlopen(url + "api/info", timeout=1.5) as r:
            if r.status == 200:
                return url
    except Exception:  # noqa: BLE001
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="drawing2fea-gui", description="Drawing2FEA desktop app")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--no-window", action="store_true", help="do not open a window (server only)")
    ap.add_argument("--selftest", action="store_true", help="run an example analysis and exit (for packaging)")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    workdir = Path(args.workdir or default_workdir())
    workdir.mkdir(parents=True, exist_ok=True)
    url = _existing_instance(workdir) if not args.port else None
    if url:  # already running: just bring up another window
        if not args.no_window:
            open_window(url, workdir)
        return 0

    app = App(workdir)
    server = serve(app, args.port)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/"
    (workdir / ".server.json").write_text(json.dumps({"port": port, "pid": os.getpid()}), encoding="utf-8")
    print(f"Drawing2FEA {__version__} running at {url}  (작업 폴더: {app.workdir})", flush=True)
    if not args.no_window:
        open_window(url, workdir)
    try:
        # Exit a while after the window is closed (no heartbeat) unless an analysis is running.
        while True:
            time.sleep(2)
            idle = time.time() - app.last_seen
            busy = any(j.status == "running" for j in app.jobs.values())
            if app.seen_client and idle > 180 and not busy and not args.no_window:
                break
    except KeyboardInterrupt:
        pass
    server.shutdown()
    return 0


def selftest() -> int:
    """Headless end-to-end check used by the packaged app's CI build."""
    import tempfile

    from ..examples import make_angle_bracket_3view_dxf
    from ..pipeline import run_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        p = make_angle_bracket_3view_dxf(Path(tmp) / "ab.dxf")
        res = run_pipeline(p, Path(tmp) / "out", overrides={"mesh": {"size": 10, "order": 1},
                                                            "analysis": {"modal": 2}}, verbose=False)
        ok = (Path(tmp) / "out" / "report.html").exists() and res["static"]["max_von_mises_MPa"] > 0
        app = App(Path(tmp) / "gui")
        server = serve(app)
        with urlopen(f"http://127.0.0.1:{server.server_address[1]}/api/info", timeout=5) as r:
            ok = ok and r.status == 200
        server.shutdown()
    msg = "SELFTEST OK" if ok else "SELFTEST FAILED"
    print(msg, flush=True)
    try:
        (Path(tempfile.gettempdir()) / "drawing2fea_selftest.txt").write_text(msg, encoding="utf-8")
    except OSError:
        pass
    return 0 if ok else 1
