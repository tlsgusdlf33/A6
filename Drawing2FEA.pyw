"""Double-click to start the Drawing2FEA desktop app (Windows, with Python installed).

``.pyw`` files run with pythonw.exe, so no console window appears.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from drawing2fea.gui.app import main  # noqa: E402

sys.exit(main())
