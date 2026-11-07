# SwiftParcel — DA3408 Assignment 5

CPU-only ETA service using FastAPI, MLflow and Ray Serve.

```text
AIOps-Assiginment5-rerun/
├── README.md
├── Analysis.md
|--AI_DISCLOSURE.md
├── eta_data.py
├── load_test.py
|-- Report.pdf
├── requirements.txt
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
├── evidence/                 # original logs and screenshots
├── data/                     # generated data and models
└── rerun/                    # new results from these commands, this is not included in this repo , but when cloned and rerun it appears
```

Add `REPORT.pdf` (or `REPORT.md`) and the AI disclosure before submission. This README assumes Part D has been renamed to `partD/smoke_eta.py`.

## Setup

Use Git, Miniforge or Conda, and at least two CPU cores on Linux, macOS or Ubuntu/WSL2. These commands create a fresh checkout at `~/AIOps-Assiginment5-rerun` and a Python 3.10 environment called `aiops-rerun`. Keep your existing `~/AIOps-Assiginment5` directory untouched. On another machine, install Miniforge first if `conda` is unavailable.

**Terminal 1 — fresh checkout and environment.**

```bash
cd ~
git clone https://github.com/Ramanagopalkodali/AIOps-Assiginment5.git AIOps-Assiginment5-rerun &&
conda create -n aiops-rerun python=3.10 pip -y &&
conda activate aiops-rerun &&
cd ~/AIOps-Assiginment5-rerun &&
python --version &&
python -m pip install -r requirements.txt "fastapi==0.115.6" "starlette==0.41.3" "ray[serve]==2.40.0" "pydantic>=2.9,<2.11" "httpx>=0.27,<0.29" "prometheus-client>=0.20" &&
python -m pip check &&
python -c 'import numpy, pandas, mlflow, ray, fastapi; print("PASS: required packages import")' &&
mkdir -p rerun/partA rerun/partB rerun/partC rerun/partD
```

The setup passes when it prints Python 3.10, `No broken requirements found` and `PASS: required packages import`. If any step fails, stop before the next section. Each later block enters the new checkout and activates `aiops-rerun`. Complete data generation before starting A.1. Keep servers running where indicated.

## Environment and data

From the repository root, activate `aiops-rerun` and record the rerun machine details. Run the following in **Terminal 2**. It also checks dependencies and saves installed versions.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
{
  uname -sm
  python --version
  python - <<'PYENV'
import os
import platform
print('OS:', platform.platform())
print('Logical CPUs:', os.cpu_count())
try:
    import psutil
    print('RAM GiB:', round(psutil.virtual_memory().total / 1024**3, 2))
except ImportError:
    print('RAM not captured automatically; record system RAM manually.')
PYENV
  python -m pip check
} 2>&1 | tee rerun/environment.txt
python -m pip freeze > rerun/packages.txt
```

In **Terminal 2**, generate the model bundles and `data/eta_sample.csv`. The output shows each model’s fresh-data MAE and the sample size.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python eta_data.py 2>&1 | tee rerun/data_generation.txt &&
python - <<'PY'
import pandas as pd
from eta_data import FEATURES
sample = pd.read_csv('data/eta_fresh.csv').loc[:, FEATURES].head(200)
assert len(sample) == 200
sample.to_csv('data/eta_sample.csv', index=False)
print(f'Created data/eta_sample.csv: {len(sample)} rows, {len(sample.columns)} features')
PY
```

New measurements go in `rerun/`. Keep the original evidence. Timings, UUIDs, process IDs and dependency warnings may differ on a rerun.

## A.1 — REST service

After generating the models, run the following in **Terminal 1**. It activates `aiops-rerun` and starts the v1 API. Leave this terminal running through the A.3 baseline and batch comparisons.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
export ETA_MODEL=data/eta_v1.joblib
python -m uvicorn partA.eta_app:app --host 127.0.0.1 --port 8000
```

In **Terminal 2**, send the valid and invalid requests, then check health and metrics. The expected statuses are 200 for valid input and 422 for invalid input.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python - <<'PY' 2>&1 | tee rerun/partA/a1_http_checks.txt
import pandas as pd
import requests
base='http://127.0.0.1:8000'
row=pd.read_csv('data/eta_sample.csv').to_dict('records')[0]
for label, item, expected in [('valid',row,200), ('invalid',{**row,'distance_km':-1},422), ('infeasible',{**row,'distance_km':151,'n_stops':21},422)]:
    response=requests.post(base+'/v1/models/eta:predict',json={'instances':[item]},headers={'X-Request-ID':'a1-check'} if label=='valid' else {},timeout=10)
    print(label,response.status_code,response.json())
    assert response.status_code==expected
    assert response.headers['X-Request-ID']==response.json()['request_id']
    if label=='valid':
        assert response.json()['request_id']=='a1-check'
print('PASS: prediction and validation checks')
health=requests.get(base+'/healthz',timeout=10)
print('Health', health.status_code, health.json())
assert health.status_code==200 and health.json()['status']=='ok'
metrics=requests.get(base+'/metrics',timeout=10)
assert metrics.status_code==200
assert 'eta_requests_total' in metrics.text and 'le="+Inf"' in metrics.text
print('Metrics 200: request counter and latency histogram present')
print('PASS: A.1 live checks')
PY
```

## A.2 — API tests

From the repository root, activate `aiops-rerun` and run the API tests in **Terminal 2**. They use TestClient and do not need a separate server. The existing suite contains 15 cases.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python -m pytest -q partA/test_eta_app.py 2>&1 | tee rerun/partA/a2_pytest.txt
```

## A.3 — Load tests

With the one-worker server still running in Terminal 1, run the baseline sweep and batch comparisons in **Terminal 2**. Each command prints a result table and saves a new JSON file.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --batch 1 --out rerun/partA/baseline.json 2>&1 | tee rerun/partA/a3_baseline.txt
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 1 --out rerun/partA/batch1.json 2>&1 | tee rerun/partA/a3_batch1.txt
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 --duration 4 --batch 100 --out rerun/partA/batch100.json 2>&1 | tee rerun/partA/a3_batch100.txt
```

**Terminal 1.** Press **Ctrl+C**, then restart with two workers.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
export ETA_MODEL=data/eta_v1.joblib
python -m uvicorn partA.eta_app:app --host 127.0.0.1 --port 8000 --workers 2
```

**Terminal 2.** Run the two-worker comparison. Each load command prints throughput and p50/p95/p99.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python load_test.py --url http://127.0.0.1:8000/v1/models/eta:predict --data data/eta_sample.csv --concurrency 1 2 4 8 16 --duration 4 --batch 1 --out rerun/partA/two_workers.json 2>&1 | tee rerun/partA/a3_two_workers.txt
```

**Terminal 1.** Press **Ctrl+C** after the comparison finishes. Keep port 8000 free for Part C.

## B.1 — MLflow registration

After generating the models and sample data, activate `aiops-rerun` and register both models from the repository root in **Terminal 2**. A fresh store creates prod → version 1 (v1) and candidate → version 2 (v2). Reuse that store for B.2 and B.3.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
export MLFLOW_TRACKING_URI="sqlite:///$PWD/rerun/partB/mlflow.db"
if [ -e rerun/partB/mlflow.db ]; then
  echo "Registry exists. Reuse it for B.2/B.3, or choose a new database for a fresh run."
else
  python -m partB.eta_mlflow 2>&1 | tee rerun/partB/b1_registration.txt
fi
```

## B.2 — Promotion gate

After B.1, run the following in **Terminal 2**. It activates `aiops-rerun`, selects the same registry and runs the six gate tests followed by the default and strict gate evaluations.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
export MLFLOW_TRACKING_URI="sqlite:///$PWD/rerun/partB/mlflow.db"
python -m pytest -q partB/test_gate.py 2>&1 | tee rerun/partB/b2_pytest.txt
python - <<'PY' 2>&1 | tee rerun/partB/gate_output.txt
import json
from partB.eta_mlflow import promotion_gate
for label, thresholds in [('Default thresholds',{}),('Strict threshold: min_rel_gain=0.60',{'min_rel_gain':0.60})]:
    print(f'\n=== {label} ===', flush=True)
    print(json.dumps(promotion_gate(**thresholds), indent=2))
PY
```

With the original bundles, the default decision was true and the 60% gain decision was false. Latencies vary.

## B.3 — Shadow comparison

From the repository root, activate `aiops-rerun` and start prod in **Terminal 3**. Leave the server running.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
export MLFLOW_TRACKING_URI="sqlite:///$PWD/rerun/partB/mlflow.db"
mlflow models serve -m 'models:/eta-model@prod' --host 127.0.0.1 --port 5101 --env-manager local
```

Start candidate in **Terminal 4**, using the same environment and registry. Leave the server running.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
export MLFLOW_TRACKING_URI="sqlite:///$PWD/rerun/partB/mlflow.db"
mlflow models serve -m 'models:/eta-model@candidate' --host 127.0.0.1 --port 5102 --env-manager local
```

After both servers report that they are ready, run the comparison in **Terminal 2**. It sends the same 200 single-row requests to both servers and prints prediction disagreement and each server’s HTTP p95.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python - <<'PYSHADOW' 2>&1 | tee rerun/partB/shadow_output.txt
from time import perf_counter
import numpy as np
import pandas as pd
import requests
from eta_data import FEATURES

rows = pd.read_csv('data/eta_fresh.csv').loc[:, FEATURES].head(200).to_dict('records')
assert len(rows) == 200
predictions = {5101: [], 5102: []}
latencies = {5101: [], 5102: []}
with requests.Session() as session:
    for row in rows:
        for port, version in ((5101, 'v1'), (5102, 'v2')):
            start = perf_counter()
            response = session.post(f'http://127.0.0.1:{port}/invocations',
                                    json={'dataframe_records': [row]}, timeout=10)
            elapsed = (perf_counter() - start) * 1000
            response.raise_for_status()
            result = response.json()['predictions']
            assert len(result) == 1 and result[0]['model_version'] == version
            eta = float(result[0]['eta_minutes'])
            assert np.isfinite(eta)
            predictions[port].append(eta)
            latencies[port].append(elapsed)
diff = np.abs(np.array(predictions[5101]) - np.array(predictions[5102]))
print('Matched requests: 200 per server')
print('Versions verified: prod=v1, candidate=v2')
print(f'Mean absolute ETA difference: {diff.mean():.3f} minutes')
print(f'Difference >10 minutes: {(diff > 10).sum()}/200 ({(diff > 10).mean():.2%})')
for port, label in ((5101, 'Prod'), (5102, 'Candidate')):
    print(f'{label} HTTP p95: {np.percentile(latencies[port], 95):.3f} ms')
print('PASS: B.3 shadow comparison')
PYSHADOW
```

**Terminals 3 and 4.** Press **Ctrl+C** after the comparison finishes.

## C.1 — Live Ray smoke check

With port 8000 free, activate `aiops-rerun` and run the following in **Terminal 2**. It starts a two-CPU Ray cluster, checks ordinary v1 routing and the v2 header override, then stops Ray automatically.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
RAY_USAGE_STATS_ENABLED=0 python -u - <<'PYC1' 2>&1 | tee rerun/partC/c1_smoke.txt
import math
import pandas as pd
import ray
import requests
from ray import serve
from eta_data import FEATURES
from partC.eta_ray import build

ray.init(address='local', num_cpus=2, include_dashboard=False)
try:
    serve.start(http_options={'host':'127.0.0.1','port':8000})
    serve.run(build(weight=0.0), name='eta')
    print(f"Ray logical CPUs: {ray.cluster_resources()['CPU']:.0f}")
    row=pd.read_csv('data/eta_sample.csv').loc[:, FEATURES].head(1).to_dict('records')[0]
    payload={'customer_id':'c1-smoke','instances':[row]}
    for headers, expected in (({},'v1'),({'X-Beta':'1'},'v2')):
        response=requests.post('http://127.0.0.1:8000/v1/models/eta:predict',
                               json=payload, headers=headers, timeout=30)
        response.raise_for_status()
        result=response.json()
        assert result['arm']==result['model_version']==expected
        assert len(result['predictions'])==1 and math.isfinite(result['predictions'][0])
        print(f"HTTP {response.status_code} arm={result['arm']} response={result}")
    print('PASS: C.1 prod routing and candidate header override')
finally:
    try:
        serve.shutdown()
    finally:
        ray.shutdown()
PYC1
```

## C.2 — Sticky canary verification

After C.1 finishes, run the existing verification script from the repository root in **Terminal 2**. A successful run prints five numbered PASS lines and a final PASS. The script starts and stops Ray automatically.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
RAY_USAGE_STATS_ENABLED=0 python -u -m partC.verify_ray 2>&1 | tee rerun/partC/verify_output.txt
```

## C.3 — Written answers

Hashing, stickiness and autoscaling explanations are in `Analysis.md` and belong in the final report.

## D.1 — Release gate

From the repository root, activate `aiops-rerun` and start the 10% canary in **Terminal 1**. Wait for READY and leave this terminal running while executing the smoke checks in Terminal 2.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
RAY_USAGE_STATS_ENABLED=0 python -u - <<'PYRAY'
import time
import ray
from ray import serve
from partC.eta_ray import build

ray.init(address='local', num_cpus=2, include_dashboard=False)
try:
    serve.start(http_options={'host': '127.0.0.1', 'port': 8000})
    serve.run(build(weight=0.1), name='eta')
    print('READY: 10% canary', flush=True)
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    serve.shutdown()
    ray.shutdown()
PYRAY
```

**Terminal 2.** After READY, check the CLI and v2. Expected exit code is 0.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python -m partD.smoke_eta --help
python -m partD.smoke_eta --url http://127.0.0.1:8000/v1/models/eta:predict --expect-version v2 --max-p95-ms 80 --force-beta
echo "Exit code: $?"
```

**Terminal 2.** Select a v1 customer and check v1. Expected exit code is 0.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
v1_customer=$(python - <<'PY'
import hashlib
for i in range(100):
    customer=f'd1-smoke-{i}'
    bucket=int.from_bytes(hashlib.sha256(customer.encode()).digest()[:8],'big')
    if bucket >= int(0.1*2**64):
        print(customer)
        break
else:
    raise SystemExit('No v1 customer found')
PY
)
python -m partD.smoke_eta --url http://127.0.0.1:8000/v1/models/eta:predict --expect-version v1 --customer-id "$v1_customer"
echo "v1 exit code should be 0: $?"
```

**Terminal 2.** Test the failure path by forcing v2 while expecting v1. Expected exit code is 1.

```bash
cd ~/AIOps-Assiginment5-rerun
conda activate aiops-rerun
unset PYTHONPATH
set -o pipefail
python -m partD.smoke_eta --url http://127.0.0.1:8000/v1/models/eta:predict --expect-version v1 --force-beta
echo "Exit code: $?"
```

**Terminal 1.** Press **Ctrl+C** when all smoke runs finish. Record the four proposed abort criteria in the report.

## Screenshots and evidence

Capture each terminal after its result table, test summary or final PASS line. For A.1 and the two-worker run, also capture the server startup. For D.1, capture both successful model checks and keep the deliberate failure check separate.

Logs already written to `rerun/` are new evidence. For another command, append `2>&1 | tee rerun/<name>.txt` to save its output. Through a pipeline, inspect `${PIPESTATUS[0]}` immediately for the Python exit code.

## Tested and not tested

Original VM evidence covers A.1 HTTP checks, 15 A.2 tests, baseline/batch/two-worker loads, registration, six gate tests, both gate decisions, shadow comparison, C.1 smoke, five C.2 checks and the original D.1 smoke checks.

The README was verified from a clean checkout on Linux aarch64 using Python 3.10. Data generation and the checks for Parts A–D completed successfully, including the expected D.1 failure exit code. The measurements discussed in the report come from the submitted result files and evidence. The clean-checkout logs are reproducible by following this README and are not included in the submission.

Original machine: Linux aarch64 VM, 4 CPUs, 7.6 GiB RAM, 4 GiB swap, Python 3.10.13. See `evidence/environment.txt`.

## Submission

Use `Module4_<rollno>.zip` with the report, required partA–partD files and reproduction instructions. Exclude models, databases, MLflow artifacts, environments, caches and OS metadata. Regenerate data/models using the commands above.

Course starter files and concepts are from https://github.com/sudarsun/DA3408. I have Documented ChatGPT assistance in the AI disclosure, following the course AI fair-use policy.

