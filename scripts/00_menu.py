#!/usr/bin/env python3
"""Menú interactivo principal para MLB Predict."""

from __future__ import annotations

import sys

from _project_root import chdir_to_project_root


def main() -> int:
    chdir_to_project_root()
    from mlb_prediction_model import main as cli_main

    return cli_main(["menu", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
