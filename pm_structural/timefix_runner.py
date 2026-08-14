from __future__ import annotations

import argparse
import importlib
import sys

import pandas as pd


def robust_to_ms(s: pd.Series) -> pd.Series:
    """Convert arbitrary pandas datetime resolution to epoch milliseconds.

    pandas 3 can preserve/choose microsecond datetime64 resolution. The old
    helper assumed nanoseconds and always divided the integer backing values by
    1e6, silently shrinking timestamps by 1000x when dtype.unit == 'us'.
    """
    x = pd.to_datetime(s, utc=True, errors="coerce")
    unit = getattr(x.dtype, "unit", "ns")
    vals = x.astype("int64")
    if unit == "ns":
        out = vals // 1_000_000
    elif unit == "us":
        out = vals // 1_000
    elif unit == "ms":
        out = vals
    elif unit == "s":
        out = vals * 1_000
    else:
        raise RuntimeError(f"unsupported pandas datetime unit: {unit!r}")
    return pd.Series(pd.array(out, dtype="Int64"), index=s.index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["strict", "diag"])
    args, rest = ap.parse_known_args()

    strict = importlib.import_module("pm_structural.obadiaha_strict")
    strict.to_ms = robust_to_ms

    if args.target == "strict":
        target = strict
    else:
        target = importlib.import_module("pm_structural.obadiaha_diag")
        target.to_ms = robust_to_ms

    # Preserve the target script's argparse contract.
    sys.argv = [getattr(target, "__file__", args.target)] + rest
    target.main()


if __name__ == "__main__":
    main()
