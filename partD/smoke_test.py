"""Run the ETA release gate against a live prediction endpoint."""

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
import requests

from eta_data import FEATURES

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)
    print(f"PASS: {message}")


def smoke_test(args):
    rows = (
        pd.read_csv(ROOT / "data" / "eta_sample.csv", nrows=2)
        .loc[:, list(FEATURES)]
        .to_dict(orient="records")
    )
    if len(rows) != 2:
        raise ValueError("Expected two sample rows")

    payload = {"customer_id": args.customer_id, "instances": rows}
    headers = {"X-Beta": "1"} if args.force_beta else {}

    with requests.Session() as session:
        response = session.post(args.url, json=payload, headers=headers, timeout=5)
        require(response.status_code == 200, "Valid batch returns HTTP 200")
        body = response.json()
        require(
            body["arm"] == body["model_version"] == args.expect_version,
            f"Arm and model version are both {args.expect_version}",
        )
        predictions = body["predictions"]
        require(
            isinstance(predictions, list) and len(predictions) == len(rows),
            "One prediction per input row",
        )
        require(
            all(type(eta) in (int, float) and math.isfinite(eta) for eta in predictions),
            "All ETA predictions are finite numbers",
        )
        require(
            all(0 < eta <= 1440 for eta in predictions),
            "All smoke fixture ETAs are positive and at most 24 hours",
        )

        invalid = session.post(
            args.url, json={**payload, "instances": []}, headers=headers, timeout=5
        )
        require(invalid.status_code == 422, "Empty batch returns HTTP 422")

        latencies = []
        for _ in range(30):
            start = perf_counter()
            response = session.post(args.url, json=payload, headers=headers, timeout=5)
            latencies.append((perf_counter() - start) * 1000)
            response.raise_for_status()
            body = response.json()
            if body["arm"] != args.expect_version or body["model_version"] != args.expect_version:
                raise ValueError("A timed request returned the wrong arm or model version")

    p95 = sorted(latencies)[math.ceil(0.95 * len(latencies)) - 1]
    print(f"HTTP p95 over {len(latencies)} requests: {p95:.2f} ms")
    require(p95 <= args.max_p95_ms, f"HTTP p95 is within {args.max_p95_ms:g} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--expect-version", required=True, choices=("v1", "v2"))
    parser.add_argument("--max-p95-ms", type=float, default=80)
    parser.add_argument("--customer-id", default="d1-smoke")
    parser.add_argument("--force-beta", action="store_true")
    args = parser.parse_args()

    try:
        if not math.isfinite(args.max_p95_ms) or args.max_p95_ms <= 0:
            raise ValueError("--max-p95-ms must be positive and finite")
        smoke_test(args)
    except (requests.RequestException, OSError, ValueError, TypeError, KeyError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"PASS: D.1 smoke test for {args.expect_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
