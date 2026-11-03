"""Verify SwiftParcel's Ray Serve canary over HTTP."""

import math
from pathlib import Path

import pandas as pd
import ray
import requests
from ray import serve

from eta_data import FEATURES
from partC.eta_ray import build

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "eta"
URL = "http://127.0.0.1:8000/v1/models/eta:predict"


def get_arm(session, customer_id, row, *, beta=False):
    headers = {"X-Beta": "1"} if beta else {}
    response = session.post(
        URL,
        json={"customer_id": customer_id, "instances": [row]},
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    arm = result["arm"]
    assert arm in {"v1", "v2"}, result
    assert result["model_version"] == arm, result
    assert len(result["predictions"]) == 1, result
    assert math.isfinite(result["predictions"][0]), result
    return arm


def main():
    row = (
        pd.read_csv(ROOT / "data" / "eta_sample.csv", nrows=1)
        .loc[:, list(FEATURES)]
        .to_dict(orient="records")[0]
    )
    customers = [f"customer-{index:04d}" for index in range(1000)]
    repeat_customers = customers[:200]

    ray.init(address="local", num_cpus=2, include_dashboard=False,
             log_to_driver=False)
    try:
        cpu_count = ray.cluster_resources()["CPU"]
        assert cpu_count == 2, cpu_count
        print(f"Ray logical CPUs: {cpu_count:.0f}")

        serve.start(http_options={"host": "127.0.0.1", "port": 8000})
        serve.run(build(weight=0.2), name=APP_NAME)

        with requests.Session() as session:
            baseline = {
                customer: get_arm(session, customer, row)
                for customer in customers
            }
            candidate_count = sum(arm == "v2" for arm in baseline.values())
            share = candidate_count / len(customers)
            assert 160 <= candidate_count <= 240, candidate_count
            print(f"PASS 1: v2 share {candidate_count}/1000 ({share:.1%}), "
                  "within 16%–24%")

            for customer in repeat_customers:
                assert get_arm(session, customer, row) == baseline[customer], customer
            print("PASS 2: 200/200 repeat requests kept their original arm")

            serve.delete(APP_NAME)
            serve.run(build(weight=0.2), name=APP_NAME)
            for customer in repeat_customers:
                assert get_arm(session, customer, row) == baseline[customer], customer
            print("PASS 3: 200/200 assignments survived delete and redeploy")

            serve.run(build(weight=0.5), name=APP_NAME)
            expanded = {
                customer: get_arm(session, customer, row)
                for customer in customers
            }
            demoted = [
                customer for customer in customers
                if baseline[customer] == "v2" and expanded[customer] == "v1"
            ]
            assert not demoted, demoted
            expanded_count = sum(arm == "v2" for arm in expanded.values())
            assert expanded_count > candidate_count, expanded_count
            print(f"PASS 4: weight 0.2 → 0.5 grew v2 from {candidate_count} "
                  f"to {expanded_count} customers, with 0 v2-to-v1 moves")

            prod_customers = [
                customer for customer in customers if expanded[customer] == "v1"
            ][:50]
            assert len(prod_customers) == 50
            for customer in prod_customers:
                assert get_arm(session, customer, row, beta=True) == "v2", customer
            print("PASS 5: X-Beta: 1 routed 50/50 v1 customers to v2")

        print("PASS: C.2 canary verification")
    finally:
        try:
            serve.shutdown()
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
