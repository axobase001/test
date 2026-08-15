from __future__ import annotations

import argparse
import importlib
import sys

from pm_structural.time_units import epoch_series_to_ms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["strict", "diag", "openmarket", "timeaudit"])
    args, rest = ap.parse_known_args()

    # Patch the frozen legacy implementation first because several audit modules
    # import its to_ms symbol at module import time.
    strict = importlib.import_module("pm_structural.obadiaha_strict")
    strict.to_ms = epoch_series_to_ms

    if args.target == "strict":
        target = strict
    elif args.target == "diag":
        target = importlib.import_module("pm_structural.obadiaha_diag")
        target.to_ms = epoch_series_to_ms
    elif args.target == "openmarket":
        target = importlib.import_module("pm_structural.openmarket_probe")
        target.to_ms = epoch_series_to_ms
    else:
        target = importlib.import_module("pm_structural.obadiaha_time_audit")
        target.to_ms = epoch_series_to_ms

    sys.argv = [getattr(target, "__file__", args.target)] + rest
    target.main()


if __name__ == "__main__":
    main()
