"""Coloca el cwd en la raíz del repo y añade esa ruta a sys.path."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def chdir_to_project_root() -> Path:
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
