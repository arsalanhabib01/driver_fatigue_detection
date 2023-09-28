import argparse
import os
import pickle

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import ast
import pandas as pd
import glob
import random
import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import precision_recall_fscore_support

HPO_EXPERIMENT_NAME = "random-forest-hyperopt-all"
EXPERIMENT_NAME = "random-forest-best-models_alldb"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

SPACE = {
    'pca': { 'n_components': scope.int(hp.quniform('n_components', 2, 80, 1))},
    'clf': { 'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
             'n_estimators': scope.int(hp.quniform('n_estimators', 50, 300, 1)),
             'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
             'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
             'random_state': 42,
             'objective': 'binary:logistic',
             'eval_metric':'merror' }
}

def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def f_unpack_dict(dct):
    """
    Unpacks all sub-dictionaries in given dictionary recursively. There should be no duplicated keys
    across all nested subdictionaries, or some instances will be lost without warning

    Parameters:
    ----------------
    dct : dictionary to unpack

    Returns:
    ----------------
    : unpacked dictionary
    """
    res = {}
    for (k, v) in dct.items():
        if isinstance(v, dict):
            res = {**res, **f_unpack_dict(v)}
        else:
            res[k] = v
    return res

def convert_run_dict(run):
    res = {}
    for (k, v) in run.items():
        if isinstance(v, str):
             res[k] = ast.literal_eval(v)
    return res

def train_and_log_model(data_path, params, participant_test=0):
    xtrain, ytrain, gtrain = load_pickle(os.path.join(data_path, "train_participant"+str(participant_test)+".pkl"))
    xtest, ytest, gtest = load_pickle(os.path.join(data_path, "test_participant"+str(participant_test)+".pkl"))

    with mlflow.start_run():
        params = space_eval(SPACE, f_unpack_dict(params))

        pipeline = make_pipeline(SimpleImputer(), PCA(**params['pca']), xgb.XGBClassifier(**params['clf'])) #objective='binary:logistic', eval_metric='merror'))
        #param_grid = {'pca__n_components': [5,10,25,30,50,75], 'xgbclassifier__n_estimators': [100,150,200,500],'xgbclassifier__max_depth': [7,9,11,13,15]}
        #pipeline.set_params(**params)
        pipeline.fit(xtrain, ytrain)

        # evaluate model on the validation and test sets
        metrics = precision_recall_fscore_support(ytest, pipeline.predict(xtest), average="weighted")
        mlflow.log_metric("test_precision", metrics[0])
        mlflow.log_metric("test_recall", metrics[1])
        mlflow.log_metric("test_f1", metrics[2])


def run(data_path, log_top):

    client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.score DESC"]
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=convert_run_dict(run.data.params))

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs( experiment_ids=experiment.experiment_id,
    run_view_type=ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.test_f1 DESC"] )[0]

    # register the best model
    mlflow.register_model( model_uri=f"runs:/{best_run.info.run_id}/model", name="fatigue_detection" )


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./data",
        help="the location where the processed data was saved."
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote."
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)
