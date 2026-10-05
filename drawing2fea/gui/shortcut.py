"""Create a double-clickable desktop icon for the Drawing2FEA app.

* Windows: ``Drawing2FEA.lnk`` on the desktop and in the Start menu (pythonw, no console window)
* macOS:   ``Drawing2FEA.app`` on the desktop (and in ~/Applications)
* Linux:   ``drawing2fea.desktop`` launcher on the desktop and in the application menu
"""

from __future__ import annotations

import os
import plistlib
import stat
import subprocess
import sys
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
APP_NAME = "Drawing2FEA"


def _package_root() -> Path:
    """Directory that must be on sys.path for ``import drawing2fea`` (repo root or site-packages)."""
    return Path(__file__).resolve().parents[2]


def launch_command() -> tuple[str, list[str]]:
    """Executable and arguments that start the GUI without a console window."""
    if getattr(sys, "frozen", False):  # PyInstaller build: the exe itself is the app
        return sys.executable, []
    exe = Path(sys.executable)
    if sys.platform.startswith("win"):
        w = exe.with_name("pythonw.exe")
        if w.exists():
            exe = w
    return str(exe), ["-m", "drawing2fea.gui"]


def _desktop() -> Path:
    if sys.platform.startswith("win"):
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "[Environment]::GetFolderPath('Desktop')"],
                                 capture_output=True, text=True, timeout=30)
            if out.stdout.strip():
                return Path(out.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
    for name in ("Desktop", "바탕 화면", "바탕화면"):
        p = Path.home() / name
        if p.is_dir():
            return p
    try:  # Linux (xdg)
        out = subprocess.run(["xdg-user-dir", "DESKTOP"], capture_output=True, text=True, timeout=5)
        if out.stdout.strip():
            return Path(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return Path.home() / "Desktop"


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def create_windows(target_dir: Path | None = None) -> list[Path]:
    exe, args = launch_command()
    workdir = str(_package_root()) if args else str(Path(exe).parent)
    targets = [target_dir or _desktop()]
    start_menu = Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    if target_dir is None and start_menu.is_dir():
        targets.append(start_menu)
    made = []
    for folder in targets:
        folder.mkdir(parents=True, exist_ok=True)
        lnk = folder / f"{APP_NAME}.lnk"
        script = (
            "$s=(New-Object -ComObject WScript.Shell).CreateShortcut(" + _ps_quote(str(lnk)) + ");"
            "$s.TargetPath=" + _ps_quote(exe) + ";"
            "$s.Arguments=" + _ps_quote(" ".join(args)) + ";"
            "$s.WorkingDirectory=" + _ps_quote(workdir) + ";"
            "$s.IconLocation=" + _ps_quote(str(ASSETS / "icon.ico") + ",0") + ";"
            "$s.Description='2D drawing -> 3D model -> FEA';$s.Save()"
        )
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                       check=True, capture_output=True, timeout=60)
        made.append(lnk)
    return made


def create_macos(target_dir: Path | None = None) -> list[Path]:
    exe, args = launch_command()
    made = []
    for folder in [target_dir or _desktop(), *([] if target_dir else [Path.home() / "Applications"])]:
        app = folder / f"{APP_NAME}.app"
        macos = app / "Contents" / "MacOS"
        res = app / "Contents" / "Resources"
        macos.mkdir(parents=True, exist_ok=True)
        res.mkdir(parents=True, exist_ok=True)
        (res / "icon.icns").write_bytes((ASSETS / "icon.icns").read_bytes())
        launcher = macos / APP_NAME
        cmd = " ".join(f'"{a}"' for a in [exe, *args])
        launcher.write_text(f'#!/bin/sh\ncd "{_package_root()}"\nexec {cmd}\n', encoding="utf-8")
        launcher.chmod(launcher.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        with open(app / "Contents" / "Info.plist", "wb") as fh:
            plistlib.dump({"CFBundleName": APP_NAME, "CFBundleDisplayName": APP_NAME,
                           "CFBundleIdentifier": "io.github.drawing2fea", "CFBundleExecutable": APP_NAME,
                           "CFBundleIconFile": "icon", "CFBundlePackageType": "APPL",
                           "CFBundleShortVersionString": "0.1.0", "LSUIElement": False}, fh)
        made.append(app)
    return made


def create_linux(target_dir: Path | None = None) -> list[Path]:
    exe, args = launch_command()
    cmd = " ".join(f'"{a}"' for a in [exe, *args])
    entry = (
        "[Desktop Entry]\nType=Application\nName=Drawing2FEA\n"
        "Comment=2D drawing -> 3D model -> FEA\n"
        f"Exec=sh -c 'cd \"{_package_root()}\" && exec {cmd}'\n"
        f"Icon={ASSETS / 'icon.png'}\nTerminal=false\nCategories=Engineering;Science;\n"
    )
    folders = [target_dir or _desktop()]
    if target_dir is None:
        folders.append(Path.home() / ".local" / "share" / "applications")
    made = []
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        f = folder / "drawing2fea.desktop"
        f.write_text(entry, encoding="utf-8")
        f.chmod(f.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        made.append(f)
    return made


def create_shortcut(target_dir: Path | None = None) -> list[Path]:
    if sys.platform.startswith("win"):
        return create_windows(target_dir)
    if sys.platform == "darwin":
        return create_macos(target_dir)
    return create_linux(target_dir)
