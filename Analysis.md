# SwiftParcel Assignment 5 — Analysis

## Setup and data generation

The setup ran on a Linux aarch64 VM with 4 logical CPUs, 7.6 GiB of RAM, and Python 3.10.13 in the `aiops` Conda environment. `python -m pip check` reported no broken requirements.

The supplied `eta_data.py` generated the training and fresh datasets and both model bundles. The load-test sample uses the first 200 rows of `eta_fresh.csv` and contains the seven input features.

| Model | Fresh-data MAE reported by the generator |
|---|---:|
| v1 | 11.95 min |
| v2 | 6.10 min |

These MAE values are from data generation. The Part B promotion gate will be measured separately.

### Setup evidence

- [Environment and dependency check](evidence/environment.txt)
- [Data and model generation output](evidence/data_generation.txt)
- [Environment setup screenshot](evidence/screenshots/emvironment_setup.png)

## A.1 — Live API verification

On the freshly generated v1 bundle, the sample request returned HTTP 200
with an ETA of 102.2 minutes and echoed the supplied request ID. A negative
distance and an infeasible delivery (151 km with 21 stops) both returned
HTTP 422 with structured validation details. For those error responses, the
generated request ID matched the value in the response header.

`/healthz` returned HTTP 200 and reported v1. `/metrics` returned HTTP 200
and exposed the request counter and latency histogram, including its
`+Inf` bucket, sum, and count.

Evidence: [A.1 live checks](evidence/partA/a1_http_checks.txt).


## A.2 — Automated API tests

All 15 tests passed in 0.95 s on the fresh setup. The suite covers valid
predictions, batch ordering and limits, invalid inputs, the infeasible delivery
rule, request ID handling, health, metrics, and a simulated inference failure.

Pytest reported one Starlette TestClient deprecation warning.

- [Pytest output](evidence/partA/a2_pytest.txt)
- [Pytest screenshot](evidence/screenshots/05_a2_pytest.png)

## A.3 — Load testing and capacity

The v1 API and load generator ran on the same Linux aarch64 VM with
4 logical CPUs and 7.6 GiB of RAM. Each test ran for 4 seconds at the
specified concurrency. Both worker sweeps used batch size 1.

| Concurrent clients | 1 worker req/s | 1 worker p95 (ms) | 1 worker p99 (ms) | 2 workers req/s | 2 workers p95 (ms) | 2 workers p99 (ms) |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 413.0 | 4.6  | 5.5  | 657.2  | 2.8  | 3.2  |
| 2  | 568.0 | 5.8  | 7.2  | 1091.5 | 3.3  | 4.3  |
| 4  | 542.5 | 12.7 | 17.1 | 1261.2 | 5.7  | 7.6  |
| 8  | 516.8 | 26.9 | 34.0 | 1347.5 | 10.9 | 15.4 |
| 16 | 490.2 | 53.4 | 62.8 | 1083.8 | 26.2 | 34.5 |

The single-worker run peaked at 568.0 requests/s with 2 concurrent clients.
Beyond that point, adding clients increased latency while throughput fell.
This is consistent with Little’s Law: when throughput stops growing,
more requests in flight correspond to a longer average time in the system.

All tested levels met the p95 ≤ 80 ms SLO and had zero errors. Adding a
second Uvicorn worker raised the highest measured throughput under the SLO
from 568.0 requests/s at concurrency 2 to 1347.5 requests/s at concurrency 8,
a 2.37× improvement in these runs. At the same concurrency of 8, p95 fell
from 26.9 ms to 10.9 ms. These figures describe the tested loads on this VM.

### Batch comparison at concurrency 1

| Batch size | Requests/s | Median request latency (ms) | Amortized median time per row (ms) |
|---:|---:|---:|---:|
| 1   | 633.0 | 1.4 | 1.4   |
| 100 | 448.2 | 2.0 | 0.020 |

Batching shares request overhead across 100 predictions. The 0.020 ms per
row is the median request latency divided by 100. Each prediction still
arrives when the full batch response returns.

The mean can hide slow requests that users still experience. The p95 and
p99 show the latency thresholds within which 95% and 99% of requests
completed, respectively.

### Evidence

- [Baseline results](partA/loadtest_results.json)
- [Batch comparison](evidence/partA/a3_batch_comparison.txt)
- [Latest two-worker results](evidence/partA/a3_two_workers.json)

## B.1 — MLflow model registration

Using MLflow 3.17.0 and a fresh SQLite tracking database, I registered
both ETA bundles under `eta-model`. Each pyfunc model was logged with
an input example, a signature, and pinned model dependencies.

| Alias | Registry version | Model bundle |
|---|---:|---|
| `prod` | 1 | `v1` |
| `candidate` | 2 | `v2` |

I loaded both models by alias and verified that each returned one
prediction per input row, with the expected bundle version in the
`model_version` column.

- [Registration and verification log](evidence/partB/b1_registration.txt)
- [Registry verification screenshot](evidence/screenshots/09_b1_registry_aliases.png)

## B.2 — Model promotion gate

All six gate tests passed in 0.86 s. The tests covered the exact threshold
boundaries, isolated failure of each criterion, and the 60% MAE gain
override. MLflow emitted one warning about missing `predict` type hints.

| Metric | `prod` (v1) | `candidate` (v2) |
|---|---:|---:|
| Overall MAE (min) | 11.9553 | 6.1049 |
| Rush-hour MAE (min) | 19.6799 | 8.5795 |
| Rain MAE (min) | 11.6227 | 6.2853 |
| p95 inference latency, default run (ms) | 1.329 | 1.123 |
| p95 inference latency, strict run (ms) | 0.994 | 0.824 |

The candidate reduced overall MAE by 48.94% and improved MAE on both
slices. Under the default thresholds, it cleared the 3% minimum overall
gain, stayed below the 10 ms p95 cap, and stayed within the 5% regression
limit on both slices. The gate returned `promote: true`.

With `min_rel_gain=0.60`, only the overall MAE criterion failed, so the
gate returned `promote: false`. Latency was measured separately for each
run using local pyfunc calls. B.3 measures latency over HTTP.

### Evidence

- [Both gate decisions](partB/gate_output.txt)
- [Pytest output](evidence/partB/b2_pytest.txt)
- [Six passing tests screenshot](evidence/screenshots/10_b2_test_pass.png)
- [Promotion gate screenshot showing both decisions](evidence/screenshots/10_b2_promotion_gate.png)


## B.3 — HTTP shadow comparison

I sent the first 200 rows from `data/eta_fresh.csv` as matched single-row
HTTP requests to `prod` (v1, port 5101) and `candidate` (v2, port 5102).
Every response contained a finite ETA and the expected model version.

| Metric | Result |
|---|---:|
| Matched requests | 200 per server |
| Mean absolute difference between ETAs | 9.484 min |
| Requests with an ETA difference greater than 10 min | 70/200 (35.00%) |
| Production HTTP p95 | 6.866 ms |
| Candidate HTTP p95 | 7.277 ms |

I would advance v2 to a controlled canary because it passed the default
B.2 gate and reduced overall MAE by 48.94%, with improvements on both
evaluated slices. Its HTTP p95 was 0.411 ms higher, and 35% of matched ETAs
differed by more than 10 minutes, so the rollout should track both latency
and ETA errors against observed delivery times. MLflow resolves the alias
when the serving process loads the model, so changing `prod` in the
registry leaves already loaded processes serving their current version.
With a rolling replacement, start and health-check each new replica before
draining an old one. With a blue-green deployment, start and verify the new
pool, switch the router to it, and let the old pool finish in-flight requests.

### Evidence

- [Shadow comparison output](partB/shadow_output.txt)
- [Shadow comparison screenshot](evidence/screenshots/11_b3_shadow_comparision.png)

## C.1 — Ray Serve canary deployment

The app deployed on Ray 2.40.0 with 2 logical CPUs. The startup log
showed two v1 replicas, one initial v2 replica, and one router. The
candidate is configured to scale from one to three replicas. At the
configured maximum, all six replica actors reserve 1.5 logical CPUs
in total.

The initial import failed because Ray Serve could not serialize the
FastAPI app with the versions installed in `aiops`. Using FastAPI
0.115.6 and Starlette 0.41.3 through a project-local `PYTHONPATH`
resolved the error.

| Request at weight 0.0 | HTTP status | Arm | ETA (min) |
|---|---:|---|---:|
| Normal request | 200 | v1 | 102.2 |
| Same request with `X-Beta: 1` | 200 | v2 | 103.9 |

The smoke check verified both inference paths and the header override.
C.2 checks the traffic split, customer stickiness, and routing behavior
across redeployments and weight changes.

### Evidence

- [C.1 smoke test log](evidence/partC/c1_smoke.txt)
- [C.1 smoke test screenshot](evidence/screenshots/12_c1_ray_smoke.png)

## C.2 — Sticky canary verification

I ran `partC/verify_ray.py` against the Ray Serve HTTP endpoint on a
local Ray cluster with two logical CPUs. The script checked the response
arm and model version on each request. All five checks passed.

| Check | Observed result |
|---|---|
| Traffic split at 20% weight | 213 of 1,000 distinct customers went to v2 (21.3%), within the required 16–24% range. |
| Repeat requests | All 200 customers kept their original arm. |
| Delete and redeploy at 20% weight | All 200 rechecked customers kept their original arm. |
| Increase weight from 20% to 50% | v2 grew from 213 to 509 of the same 1,000 customers. Zero customers moved from v2 to v1. |
| `X-Beta: 1` override | All 50 tested customers assigned to v1 were routed to v2 for the override request. |

The redeployment check shows that the tested assignments can be recovered
from the customer IDs and weight after the application is deleted and
recreated. The weight increase expanded the v2 group while retaining
everyone already assigned to it.

### Evidence

- [Verification output](partC/verify_output.txt)
- [Verification screenshot](evidence/screenshots/13_c2_ray_verification.png)

## C.3 — Canary routing and autoscaling reasoning

Python's built-in `hash()` is salted per process. The same customer ID can therefore receive a different hash after a restart or on another replica, breaking stable canary assignment. Our router uses SHA-256 to produce a repeatable bucket for each customer ID.

Stickiness gives a customer consistent ETA behavior across repeated requests. It also keeps each customer in one experiment arm, so comparisons between v1 and v2 are easier to interpret.

The autoscaling target of two ongoing requests per replica aims to add capacity before queues grow large; the maximum of four ongoing requests limits queueing at a replica. A five-second upscale delay responds to sustained demand, while a 30-second downscale delay avoids removing capacity during brief dips. The candidate's one-to-three-replica range keeps v2 available while limiting its CPU use. These settings are configuration choices; C.2 verified routing behavior, not autoscaling performance.

## D.1 — Release gate and rollback policy

We ran the smoke test against both models with the canary configured at
10%. The v1 request used a customer assigned to v1 by the router. The v2
request used the `X-Beta: 1` override to reach the candidate directly.

Both runs passed all seven checks: HTTP 200 for a valid batch, the
expected arm and model version, one prediction per input, finite numeric
predictions, positive fixture ETAs of at most 24 hours, HTTP 422 for an
empty batch, and HTTP p95 within the 80 ms limit.

| Model | Timed requests | HTTP p95 | Limit | Result |
| --- | ---: | ---: | ---: | --- |
| v1 | 30 | 17.16 ms | 80 ms | PASS |
| v2 | 30 | 12.79 ms | 80 ms | PASS |

Evidence: [v1 smoke log](evidence/partD/d1_smoke_v1.txt) and
[v2 smoke log](evidence/partD/d1_smoke_v2.txt).

### Abort criteria for the 10% rollout

Set the canary weight back to 0% if any of these criteria is met.
Exclude requests using the `X-Beta` override from rollout measurements.

| Signal | Abort threshold | Observation window |
| --- | --- | --- |
| Candidate availability | More than 1% of v2 requests return HTTP 5xx or time out | Rolling 5 minutes with at least 100 v2 requests |
| Candidate latency | v2 HTTP p95 exceeds 80 ms | Rolling 5 minutes with at least 100 successful v2 requests |
| Prediction sanity | Any successful v2 response contains a nonfinite or nonpositive ETA, or returns a different number of predictions than inputs | Rolling 5 minutes |
| Prediction quality | v2 mean absolute error exceeds v1 mean absolute error by more than 10% on the same deliveries | Rolling 24 hours with at least 100 deliveries with paired predictions and known actual ETAs |

Keep the canary at 10% while collecting enough data to evaluate the
criteria. After a rollback, verify that ordinary requests route to v1.
Disable the beta override as well if v2 must receive zero traffic.

The local smoke runs verified the release gate. The rollback criteria
are proposed monitoring rules for the rollout.

[Terminal screenshot showing both smoke tests passing](evidence/screenshots/14_d1_smoke.png)
