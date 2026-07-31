#!/usr/bin/env python3
"""Compare a candidate metrics JSON against baseline condition files and print success criteria.

Example:
  python experiments/compare_efficiency_criteria.py \\
    --candidate result/metrics_B4.json \\
    --baseline B1=result/metrics_B1.json B3=result/metrics_B3.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from Topodim.utils.efficiency_metrics import (
    load_metrics_summary,
    evaluate_success_criteria,
    print_success_criteria,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate B0–B5 success criteria")
    parser.add_argument("--candidate", required=True, help="Candidate metrics JSON path")
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="Baselines as NAME=PATH (e.g. B1=result/metrics_B1.json)",
    )
    parser.add_argument("--out", default="", help="Optional path to write verdicts JSON")
    args = parser.parse_args()

    candidate = load_metrics_summary(args.candidate)
    baselines = {}
    for item in args.baseline:
        name, path = item.split("=", 1)
        baselines[name.strip()] = load_metrics_summary(path.strip())

    verdicts = evaluate_success_criteria(candidate, baselines)
    print_success_criteria(verdicts)

    if args.out:
        Path(args.out).write_text(json.dumps(verdicts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")

    n_fail = sum(1 for v in verdicts if v["status"] == "FAIL")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
