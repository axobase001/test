from __future__ import annotations

import argparse

from main_sequence import sol1h_core_audited as sol


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args()

    sol.runner.START = args.start
    sol.runner.END = args.end
    sol.configure()
    sol.runner.main()
    sol.audit()


if __name__ == "__main__":
    main()
