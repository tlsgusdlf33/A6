# PyInstaller build of the Drawing2FEA desktop app.
#
#   pip install pyinstaller
#   pyinstaller packaging/drawing2fea.spec --noconfirm
#
# Output: dist/Drawing2FEA/Drawing2FEA(.exe)   (macOS: dist/Drawing2FEA.app)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent
ASSETS = ROOT / "drawing2fea" / "gui" / "assets"
icon = str(ASSETS / ("icon.ico" if sys.platform.startswith("win") else "icon.icns"))

datas = collect_data_files("drawing2fea") + collect_data_files("ezdxf")
binaries = collect_dynamic_libs("shapely") + collect_dynamic_libs("triangle")
hidden = collect_submodules("ezdxf") + collect_submodules("drawing2fea") + ["triangle", "shapely"]
try:
    import pyamg  # noqa: F401  optional accelerator for large models

    hidden += collect_submodules("pyamg")
except ImportError:
    pass

a = Analysis(
    [str(ROOT / "packaging" / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "PyQt5", "PySide6", "IPython", "pytest", "pypardiso", "mkl", "playwright"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Drawing2FEA", icon=icon,
          console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="Drawing2FEA", upx=False)
if sys.platform == "darwin":
    app = BUNDLE(coll, name="Drawing2FEA.app", icon=icon, bundle_identifier="io.github.drawing2fea")
