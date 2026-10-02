#!/usr/bin/env python3
"""Descarga partidos y stats desde la MLB Stats API (como scripts/01_download.py de NHL)."""

from __future__ import annotations

import sys

from _project_root import chdir_to_project_root


def main() -> int:
    chdir_to_project_root()
    from mlb_prediction_model import main as cli_main

    return cli_main(["download", *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
