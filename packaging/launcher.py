"""Entry point of the packaged desktop app (PyInstaller)."""

import multiprocessing
import sys

from drawing2fea.gui.app import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
