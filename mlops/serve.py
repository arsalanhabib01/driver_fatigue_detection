import argparse
from http import HTTPStatus
from typing import Dict

import pandas as pd
import ray
from fastapi import FastAPI
from ray import serve
from ray.train.torch import TorchPredictor
from starlette.requests import Request
import numpy as np
import evaluate, predict
from config import MLFLOW_TRACKING_URI, mlflow

# Define application
app = FastAPI(
    title="Fatigue Model service",
    description="Detects fatigue event for an individual",
    version="0.1",
)


@serve.deployment(route_prefix="/", num_replicas="1", ray_actor_options={"num_cpus": 8, "num_gpus": 0})
@serve.ingress(app)
class ModelDeployment:
    def __init__(self, run_id: str, threshold: int = 0.9):
        """Initialize the model."""
        self.run_id = run_id
        self.threshold = threshold
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  # so workers have access to model registry
        best_checkpoint = predict.get_best_checkpoint(run_id=run_id)
        self.predictor = TorchPredictor.from_checkpoint(best_checkpoint)
        self.preprocessor = self.predictor.get_preprocessor()

    @app.get("/")
    def _index(self) -> Dict:
        """Health check."""
        response = {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": {},
        }
        return response

    @app.get("/run_id/")
    def _run_id(self) -> Dict:
        """Get the run ID."""
        return {"run_id": self.run_id}

    @app.post("/evaluate/")
    async def _evaluate(self, request: Request) -> Dict:
        data = await request.json()
        results = evaluate.evaluate(run_id=self.run_id, dataset_loc=data.get("dataset"))
        return {"results": results}

    @app.post("/predict/")
    async def _predict(self, request: Request) -> Dict:
        # Get prediction
        data = await request.json()
        features = data.get("features", "")
        if len(features)>0:
            features = np.array(features, dtype=np.float64)
            print(features.shape, features.dtype)
        ds = pd.DataFrame({'groups': np.array([0]), 'targets': np.array([-1])})
        ds['id'] = ds.index.values.tolist()
        ds['features'] = [features]
        results = predict.predict_with_proba(data=ds, predictor=self.predictor)

        # Apply custom logic
        for i, result in enumerate(results):
            pred = result["prediction"]
            prob = result["probabilities"]
            if prob[pred] < self.threshold:
                results[i]["prediction"] = "undetermined"

        return {"results": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", help="run ID to use for serving.")
    parser.add_argument("--threshold", type=float, default=0.8, help="threshold for `undetermined` class.")
    args = parser.parse_args()
    ray.init()
    serve.run(ModelDeployment.bind(run_id=args.run_id, threshold=args.threshold))