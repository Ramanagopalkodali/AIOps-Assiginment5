"""Register the SwiftParcel v1 and v2 ETA models in MLflow."""

import os
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

MODEL_NAME = "eta-model"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


class EtaPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        bundle = joblib.load(context.artifacts["bundle"])
        self.model = bundle["model"]
        self.feature_columns = list(bundle["feature_columns"])
        self.model_version = str(bundle["version"])

    def predict(self, context, model_input, params=None):
        features = model_input.loc[:, self.feature_columns]
        predictions = self.model.predict(features)
        return pd.DataFrame({
            "eta_minutes": [round(float(eta), 1) for eta in predictions],
            "model_version": self.model_version,
        })


def register():
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("swiftparcel-eta")
    client = MlflowClient()

    sample = pd.read_csv(DATA / "eta_sample.csv").head(3)
    packages = ("mlflow", "cloudpickle", "joblib", "numpy", "pandas",
                "scikit-learn", "scipy")
    requirements = [f"{package}=={version(package)}" for package in packages]

    for alias, label in (("prod", "v1"), ("candidate", "v2")):
        bundle_path = DATA / f"eta_{label}.joblib"
        bundle = joblib.load(bundle_path)
        if str(bundle["version"]) != label:
            raise ValueError(f"{bundle_path} does not contain model {label}")

        example = sample.loc[:, list(bundle["feature_columns"])]
        output = pd.DataFrame({
            "eta_minutes": [round(float(eta), 1)
                            for eta in bundle["model"].predict(example)],
            "model_version": label,
        })

        with mlflow.start_run(run_name=f"register-{label}"):
            info = mlflow.pyfunc.log_model(
                artifact_path=f"eta-{label}",
                python_model=EtaPyfunc(),
                artifacts={"bundle": str(bundle_path)},
                signature=infer_signature(example, output),
                input_example=example,
                pip_requirements=requirements,
            )
            registered = mlflow.register_model(info.model_uri, MODEL_NAME)

        client.set_model_version_tag(
            MODEL_NAME, registered.version, "bundle_version", label
        )
        client.set_registered_model_alias(MODEL_NAME, alias, registered.version)

    print(f"{'alias':<12} {'registry_version':<18} bundle_version")
    for alias, expected in (("prod", "v1"), ("candidate", "v2")):
        registered = client.get_model_version_by_alias(MODEL_NAME, alias)
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
        columns = model.metadata.get_input_schema().input_names()
        result = model.predict(sample.loc[:, columns])

        assert list(result.columns) == ["eta_minutes", "model_version"]
        assert len(result) == len(sample)
        assert result["model_version"].eq(expected).all()
        assert result["eta_minutes"].notna().all()

        print(f"{alias:<12} {str(registered.version):<18} {expected}")

    print("PASS: both aliases loaded and returned ETA predictions.")
    
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0,
           max_slice_regression=0.05):
    """Evaluate candidate metrics against the promotion thresholds."""
    criteria = {
        "overall_mae": (
            prod["mae"] > 0
            and cand["mae"] <= prod["mae"] * (1 - min_rel_gain)
        ),
        "p95_latency": cand["p95_ms"] <= max_p95_ms,
        "rush_mae": cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression),
        "rain_mae": cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def promotion_gate(**thresholds):
    """Load both aliases, evaluate them on fresh data, and decide."""
    from time import perf_counter

    import numpy as np
    from eta_data import FEATURES, make_data

    uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)

    fresh = make_data(2000, seed=99)
    features = fresh.loc[:, list(FEATURES)]
    target_columns = fresh.columns.difference(features.columns)
    if len(target_columns) != 1:
        raise ValueError(f"Expected one ETA target column, found {list(target_columns)}")
    actual = fresh[target_columns[0]].to_numpy()
    rush = features["hour"].isin([7, 8, 9, 16, 17, 18, 19]).to_numpy()
    rain = features["rain"].eq(1).to_numpy()
    if not rush.any() or not rain.any():
        raise ValueError("Fresh evaluation data must contain rush-hour and rain rows")

    def evaluate(alias):
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
        predicted = model.predict(features)["eta_minutes"].to_numpy()
        errors = np.abs(predicted - actual)

        durations_ms = []
        for index in range(200):
            row = features.iloc[index:index + 1]
            start = perf_counter()
            model.predict(row)
            durations_ms.append((perf_counter() - start) * 1000)

        return {
            "mae": float(errors.mean()),
            "mae_rush": float(errors[rush].mean()),
            "mae_rain": float(errors[rain].mean()),
            "p95_ms": float(np.percentile(durations_ms, 95)),
        }

    prod = evaluate("prod")
    cand = evaluate("candidate")
    return {"prod": prod, "candidate": cand, **decide(prod, cand, **thresholds)}


if __name__ == "__main__":
    register()
