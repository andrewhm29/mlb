#!/usr/bin/env python3
"""
Entrypoint simple para abrir el menu principal del predictor MLB.

Uso:
    python3 scripts/00_menu.py
"""

import sys
from pathlib import Path


def main() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from mlb_prediction_model import main as mlb_menu_main

    mlb_menu_main()


if __name__ == "__main__":
    main()
