```bash
AIOps-Assiginment5/
├── README.md
├── Analysis.md
├── AI_disclosure.md
├── requirements.txt
├── .gitignore
├── eta_data.py                   # supplied generator
├── load_test.py                  # supplied load generator
├── partA/
│   ├── eta_app.py
│   ├── test_eta_app.py
│   └── loadtest_results.json
├── partB/
│   ├── eta_mlflow.py
│   ├── test_gate.py
│   ├── gate_output.txt
│   └── shadow_output.txt
├── partC/
│   ├── eta_ray.py
│   ├── verify_ray.py
│   └── verify_output.txt
├── partD/
│   └── smoke_eta.py
├── evidence/                     # supporting outputs for the working repo
│   ├── environment.txt
│   ├── partA/                    # pytest output, batch comparison, worker comparison
│   ├── partB/                    # registry alias table and supporting logs
│   └── partD/                    # smoke output and exit status
└── data/                         # generated locally, excluded from submission ZIP
    ├── eta_train.csv
    ├── eta_fresh.csv
    ├── eta_sample.csv
    ├── eta_v1.joblib
    └── eta_v2.joblib
```

## Setup and data generation

The SwiftParcel assignment uses Python 3.10+ and runs on CPU. The commands below assume the `aiops` Conda environment is available.

```bash
conda activate aiops
python -m pip install -r requirements.txt
python -m pip check
python eta_data.py
```

Generate the 200-row sample used by the Part A load tests:

```bash
python - <<'PY'
import pandas as pd
from eta_data import FEATURES

sample = pd.read_csv("data/eta_fresh.csv").loc[:, FEATURES].head(200)
assert len(sample) == 200
sample.to_csv("data/eta_sample.csv", index=False)
print(f"Created data/eta_sample.csv: {len(sample)} rows, {len(sample.columns)} features")
PY

ls -lh data/
```

The `data/` directory should contain `eta_train.csv`, `eta_fresh.csv`, `eta_sample.csv`, `eta_v1.joblib`, and `eta_v2.joblib`. The sample contains 200 rows and the seven input features in the order defined by `eta_data.FEATURES`.


## A.1 — Run the ETA API

After generating the model bundles, run from the repository root:

```bash
conda activate aiops
ETA_MODEL=data/eta_v1.joblib python -m uvicorn partA.eta_app:app \
  --host 127.0.0.1 --port 8000
```

Keep the server running. In a second terminal, check health, submit a prediction,
and inspect the metrics:

```bash
curl -fsS http://127.0.0.1:8000/healthz

curl -fsS -X POST 'http://127.0.0.1:8000/v1/models/eta:predict' \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: a1-check' \
  -d '{"instances":[{"distance_km":12.5,"n_stops":3,"hour":11,"weekday":2,"traffic_index":0.35,"rain":0,"weight_kg":4.0}]}'

curl -fsS http://127.0.0.1:8000/metrics
```

The prediction response contains one ETA, `model_version` set to `v1`, and
`request_id` set to `a1-check`. See `evidence/partA/a1_http_checks.txt`
for the recorded live checks.

## A.2 — Run the API tests

Generate the model bundles using the setup instructions above, then run from the
repository root in the `aiops` environment. The tests use FastAPI's
`TestClient` and can run without a Uvicorn server.

```bash
conda activate aiops
mkdir -p evidence/partA
set -o pipefail
python -m pytest -q partA/test_eta_app.py 2>&1 | tee evidence/partA/a2_pytest.txt
```

## A.3 — Benchmark the v1 API

Start the v1 server as in A.1 with one Uvicorn worker. In a second terminal,
run the baseline sweep and batch comparison from the repository root:

```bash
mkdir -p evidence/partA
set -o pipefail
url='http://127.0.0.1:8000/v1/models/eta:predict'

python load_test.py --url "$url" --data data/eta_sample.csv \
  --concurrency 1 2 4 8 16 --duration 4 --batch 1 \
  --out partA/loadtest_results.json

for batch in 1 100; do
  python load_test.py --url "$url" --data data/eta_sample.csv \
    --concurrency 1 --duration 4 --batch "$batch" \
    --out "evidence/partA/a3_batch_${batch}.json" || break
done
```

Stop the server, restart it with `--workers 2`, and repeat the baseline
command with `--out evidence/partA/a3_two_workers.json`.

## B.1 — Register the ETA models in MLflow

From the repository root, activate the `aiops` environment. Generate
the model bundles and `data/eta_sample.csv` using the setup instructions
above, then run:

```bash
conda activate aiops
mkdir -p .local/partB evidence/partB
export MLFLOW_TRACKING_URI="sqlite:///$PWD/.local/partB/mlflow_b1_fresh.db"
set -o pipefail

if [ -e .local/partB/mlflow_b1_fresh.db ]; then
  echo "Registry already exists. Reuse it for B.2 and B.3."
else
  python -m partB.eta_mlflow 2>&1 | tee evidence/partB/b1_registration.txt
fi
```

On a fresh registry, `prod` should resolve to version 1 (`v1`) and
`candidate` to version 2 (`v2`). Successful registration ends with
`PASS: both aliases loaded and returned ETA predictions.` Reuse the same
tracking URI for B.2 and B.3. Running registration again against an existing
registry creates new numbered versions.

## B.2 — Run the promotion gate

Run from the repository root after completing B.1. Reuse the same SQLite
registry so the gate can load `eta-model@prod` and `eta-model@candidate`.

```bash
conda activate aiops
export MLFLOW_TRACKING_URI="sqlite:///$PWD/.local/partB/mlflow_b1_fresh.db"
mkdir -p evidence/partB
set -o pipefail

python -m pytest -q partB/test_gate.py 2>&1 | tee evidence/partB/b2_pytest.txt
```

Once all six tests pass, run both gate configurations:

```bash
python - <<'PY' 2>&1 | tee partB/gate_output.txt
import json
from partB.eta_mlflow import promotion_gate

for label, overrides in (
    ("Default thresholds", {}),
    ("Strict threshold: min_rel_gain=0.60", {"min_rel_gain": 0.60}),
):
    print(f"\n=== {label} ===", flush=True)
    print(json.dumps(promotion_gate(**overrides), indent=2), flush=True)
PY
```

The gate evaluates 2,000 fresh rows generated with seed 99 and measures
p95 over 200 single-row pyfunc predictions per model for each configuration.
Both decisions are saved in `partB/gate_output.txt`.

## B.3 — Compare the models over HTTP

Use the SQLite registry from B.1 and the `aiops` Conda environment.
Run all commands from the repository root.

In Terminal 1, start the production model:

```bash
conda activate aiops
export MLFLOW_TRACKING_URI="sqlite:///$PWD/.local/partB/mlflow_b1_fresh.db"
mlflow models serve --model-uri 'models:/eta-model@prod' \
  --host 127.0.0.1 --port 5101 --env-manager local
```

In Terminal 2, start the candidate model:

```bash
conda activate aiops
export MLFLOW_TRACKING_URI="sqlite:///$PWD/.local/partB/mlflow_b1_fresh.db"
mlflow models serve --model-uri 'models:/eta-model@candidate' \
  --host 127.0.0.1 --port 5102 --env-manager local
```

In Terminal 3, wait until both servers respond to `/ping`, then send the same
200 single-row requests to both `/invocations` endpoints:

```bash
conda activate aiops
set -o pipefail

curl -fsS http://127.0.0.1:5101/ping &&
curl -fsS http://127.0.0.1:5102/ping &&
python - <<'PY' 2>&1 | tee partB/shadow_output.txt
from time import perf_counter

import numpy as np
import pandas as pd
import requests

from eta_data import FEATURES

rows = (
    pd.read_csv("data/eta_fresh.csv", nrows=200)
    .loc[:, list(FEATURES)]
    .to_dict(orient="records")
)
assert len(rows) == 200

def predict(session, port, expected_version, payload):
    start = perf_counter()
    response = session.post(
        f"http://127.0.0.1:{port}/invocations", json=payload, timeout=10
    )
    elapsed_ms = (perf_counter() - start) * 1000
    response.raise_for_status()
    predictions = response.json()["predictions"]
    assert len(predictions) == 1, predictions
    result = predictions[0]
    assert result["model_version"] == expected_version, result
    eta = float(result["eta_minutes"])
    assert np.isfinite(eta), result
    return eta, elapsed_ms

differences, prod_ms, candidate_ms = [], [], []
with requests.Session() as session:
    for row in rows:
        payload = {"dataframe_records": [row]}
        prod_eta, prod_time = predict(session, 5101, "v1", payload)
        candidate_eta, candidate_time = predict(session, 5102, "v2", payload)
        differences.append(abs(prod_eta - candidate_eta))
        prod_ms.append(prod_time)
        candidate_ms.append(candidate_time)

diff = np.asarray(differences)
over_10 = np.count_nonzero(diff > 10)
print(f"Matched requests: {len(diff)} per server")
print("Versions verified: prod=v1, candidate=v2")
print(f"Mean absolute ETA difference: {diff.mean():.3f} minutes")
print(f"Difference >10 minutes: {over_10}/{len(diff)} ({100 * over_10 / len(diff):.2f}%)")
print(f"Prod HTTP p95: {np.percentile(prod_ms, 95):.3f} ms")
print(f"Candidate HTTP p95: {np.percentile(candidate_ms, 95):.3f} ms")
print("PASS: B.3 shadow comparison")
PY
```

After the `PASS` line appears, stop both servers with Ctrl+C.

## C.1 — Build the Ray Serve canary

From the repository root, activate `aiops` and install the compatibility
pair used with Ray 2.40.0. The project-local directory lets Part C use
these versions alongside the dependencies installed in `aiops`.

```bash
conda activate aiops
python -m pip install --target "$PWD/.local/partC/serve-deps" --no-deps \
  'fastapi==0.115.6' 'starlette==0.41.3'

PYTHONPATH="$PWD/.local/partC/serve-deps${PYTHONPATH:+:$PYTHONPATH}" \
python -c 'from partC.eta_ray import build; build(weight=0.0); print("PASS: C.1 app builds")'
```

Use the same `PYTHONPATH` prefix for the C.2 verification command. The
canary accepts `customer_id` and `instances` at
`POST /v1/models/eta:predict`. The response reports the selected `arm`
and `model_version`. With `weight=0.0`, a normal request goes to v1 and
`X-Beta: 1` routes the request to v2.

## C.2 — Verify the sticky canary

Run from the repository root after completing the C.1 dependency setup.
Make sure port 8000 is free. The verification script starts a local Ray
cluster with two logical CPUs and shuts it down after the checks.

```bash
conda activate aiops
set -o pipefail

PYTHONPATH="$PWD/.local/partC/serve-deps${PYTHONPATH:+:$PYTHONPATH}" \
RAY_USAGE_STATS_ENABLED=0 \
python -u -m partC.verify_ray 2>&1 | tee partC/verify_output.txt
```

A successful run prints five numbered `PASS` lines, followed by
`PASS: C.2 canary verification`. The required submission evidence is
`partC/verify_output.txt`.

### C.3 — Design reasoning

See [Analysis.md](Analysis.md#c3--canary-routing-and-autoscaling-reasoning) for the hash, stickiness, and autoscaling parameter explanations. No additional command is required.

## Part D — Release gate

### D.1 — Run the smoke test

From the repository root, activate the `aiops` environment. Start the
Part C Ray Serve app on `127.0.0.1:8000` with `build(weight=0.10)` and
leave it running. In a second terminal, activate the same environment
and run:

```bash
set -o pipefail
mkdir -p evidence/partD
url='http://127.0.0.1:8000/v1/models/eta:predict'

# Select a customer assigned to v1 at 10% canary weight.
v1_customer=$(python - <<'PY'
import hashlib

for i in range(100):
    customer = f"d1-smoke-{i}"
    bucket = int.from_bytes(hashlib.sha256(customer.encode()).digest()[:8], "big")
    if bucket >= int(0.10 * 2**64):
        print(customer)
        break
else:
    raise SystemExit("Could not find a v1 customer")
PY
)

python -u -m partD.smoke_test --url "$url" --expect-version v1 \
  --customer-id "$v1_customer" 2>&1 | tee evidence/partD/d1_smoke_v1.txt

python -u -m partD.smoke_test --url "$url" --expect-version v2 \
  --force-beta 2>&1 | tee evidence/partD/d1_smoke_v2.txt
```

The v2 command sends `X-Beta: 1` to exercise the candidate directly.
The smoke test defaults to an HTTP p95 limit of 80 ms, which can be
changed with `--max-p95-ms`. It exits `0` on success and `1` on a failed
check. See [Analysis.md](Analysis.md) for the measured results and
rollback criteria.

# Module 4 assignment — starter (SwiftParcel ETA service)

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python eta_data.py            # writes data/eta_train.csv, data/eta_fresh.csv, data/eta_v1.joblib, data/eta_v2.joblib
    python -c "import pandas as pd, eta_data; pd.read_csv('data/eta_fresh.csv')[eta_data.FEATURES].head(200).to_csv('data/eta_sample.csv', index=False)"

Load generator:

    python load_test.py --url http://localhost:8000/v1/models/eta:predict --data data/eta_sample.csv \
           --concurrency 1 2 4 8 16 --duration 4 --out loadtest_results.json

See the assignment document for tasks, interface contract and rubric.


