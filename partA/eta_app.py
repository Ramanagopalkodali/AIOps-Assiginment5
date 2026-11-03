"""SwiftParcel ETA REST service (Part A.1).

Run from the repository root with ``uvicorn partA.eta_app:app --port 8000``.
Set ETA_MODEL to select another model bundle before starting the process.
"""

import logging
import os
import time
import uuid
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, ConfigDict, model_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

# Load exactly once when the module is imported, never inside a request.
bundle = joblib.load(os.getenv("ETA_MODEL", "data/eta_v1.joblib"))
model = bundle["model"]
feature_columns = bundle["feature_columns"]
model_version = bundle["version"]


class Delivery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distance_km: float = Field(ge=0.1, le=500, allow_inf_nan=False)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1, allow_inf_nan=False)
    rain: Literal[0, 1]
    weight_kg: float = Field(gt=0, le=1000, allow_inf_nan=False)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance and stops are too high")
        return self


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[Delivery] = Field(min_length=1, max_length=500)


app = FastAPI(title="SwiftParcel ETA")
REQUESTS = Counter("eta_requests_total", "HTTP requests", ["path", "status"])
LATENCY = Histogram("eta_request_latency_seconds", "HTTP request duration in seconds")


def request_id(request: Request) -> str:
    return request.state.request_id


def error_response(request: Request, status: int, error: str, details: list[dict]) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": error, "details": details, "request_id": request_id(request)},
    )


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    supplied_id = request.headers.get("X-Request-ID")
    request.state.request_id = supplied_id if supplied_id is not None else str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request failure")
        response = error_response(request, 500, "internal_error", [{"field": "server", "msg": "Internal server error"}])
    response.headers["X-Request-ID"] = request_id(request)
    if request.url.path != "/metrics":
        REQUESTS.labels(path=request.url.path, status=str(response.status_code)).inc()
        LATENCY.observe(time.perf_counter() - started)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = []
    for item in exc.errors():
        parts = [part for part in item["loc"] if part != "body"]
        field = "".join(f"[{p}]" if isinstance(p, int) else (f".{p}" if i else str(p))
                        for i, p in enumerate(parts))
        details.append({"field": field or "body", "msg": str(item["msg"])})
    return error_response(request, 422, "validation_failed", details)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException):
    return error_response(request, exc.status_code, "http_error", [{"field": "request", "msg": str(exc.detail)}])


@app.get("/healthz")
def healthz(request: Request):
    return {"status": "ok", "model_version": model_version, "request_id": request_id(request)}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/models/eta:predict")
def predict(payload: PredictionRequest, request: Request):
    rows = [row.model_dump() for row in payload.instances]
    frame = pd.DataFrame.from_records(rows, columns=feature_columns)
    try:
        values = model.predict(frame)
        predictions = [round(float(value), 1) for value in values]
    except Exception:
        logger.exception("ETA inference failed")
        return error_response(request, 500, "inference_failed", [{"field": "server", "msg": "Inference failed"}])
    return {"predictions": predictions, "model_version": model_version, "request_id": request_id(request)}
