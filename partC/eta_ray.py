"""Ray Serve application with a sticky customer canary — Part C.1.

Course reference: Lecture 4c Ray Serve exercise, sections 0 and 2–4.
Relevant patterns include loading a model once per replica, composing a FastAPI
ingress router with model deployments, fractional CPUs and autoscaling settings.

The lecture uses random routing per request. This assignment requires stable
customer routing, implemented with SHA-256, plus the X-Beta override.
ETA-specific implementation developed with ChatGPT assistance.

Source: https://github.com/sudarsun/DA3408/blob/main/Lecture%2010%20--%20Ray_Serve_Exercise.ipynb
"""

"""Ray Serve deployment for the SwiftParcel sticky canary."""

import hashlib
import time
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
api = FastAPI()


class PredictionRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: list[dict] = Field(min_length=1, max_length=500)


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
class EtaModel:
    def __init__(self, bundle_path: str, extra_work_ms: float):
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.version = str(bundle["version"])
        self.extra_work_seconds = extra_work_ms / 1000

    def predict(self, instances: list[dict]) -> dict:
        if self.extra_work_seconds:
            time.sleep(self.extra_work_seconds)
        features = pd.DataFrame(instances).loc[:, self.feature_columns]
        return {
            "predictions": [round(float(eta), 1) for eta in self.model.predict(features)],
            "model_version": self.version,
        }


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
@serve.ingress(api)
class EtaRouter:
    def __init__(self, prod, candidate, weight: float):
        self.prod = prod
        self.candidate = candidate
        self.cutoff = int(weight * 2**64)

    @api.post("/v1/models/eta:predict")
    async def predict(
        self, request: PredictionRequest, x_beta: str | None = Header(default=None)
    ):
        digest = hashlib.sha256(request.customer_id.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big")
        use_candidate = x_beta == "1" or bucket < self.cutoff

        model = self.candidate if use_candidate else self.prod
        result = await model.predict.remote(request.instances)
        return {**result, "arm": "v2" if use_candidate else "v1"}


def build(weight=0.1, extra_work_ms=0.0):
    """Build the app. Weight is the fraction of customer IDs assigned to v2."""
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    if not 0 <= extra_work_ms < float("inf"):
        raise ValueError("extra_work_ms must be finite and nonnegative")

    prod = EtaModel.options(name="EtaProd", num_replicas=2).bind(
        str(DATA_DIR / "eta_v1.joblib"), extra_work_ms
    )
    candidate = EtaModel.options(
        name="EtaCandidate",
        max_ongoing_requests=4,
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 5,
            "downscale_delay_s": 30,
        },
    ).bind(str(DATA_DIR / "eta_v2.joblib"), extra_work_ms)
    return EtaRouter.bind(prod, candidate, weight)
