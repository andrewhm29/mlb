#!/usr/bin/env python3
"""Predice el calendario MLB para una fecha (igual que NHL: scripts/05_predict.py)."""

from __future__ import annotations

import sys

from _project_root import chdir_to_project_root


def main() -> int:
    chdir_to_project_root()
    from mlb_prediction_model import main as cli_main

    return cli_main(["predict", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
