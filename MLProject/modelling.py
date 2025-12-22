import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#dagshub.init(repo_owner='levinafauziah', repo_name='House-Rent-Prediction', mlflow=True)

tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
if tracking_uri:
    mlflow.set_tracking_uri(tracking_uri)

print(f"Tracking URI: {mlflow.get_tracking_uri()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=115)
    parser.add_argument("--max_depth", type=int, default=22)
    args = parser.parse_args()

    # Load Data
    df = pd.read_csv('dataset_preprocessed/data_clean.csv')
    
    X = df.drop('rent', axis=1)
    y = df['rent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    with mlflow.start_run():
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        rf.fit(X_train, y_train)

        predictions = rf.predict(X_test)

        mlflow.log_params(vars(args))
        mlflow.log_metric("mse", mean_squared_error(y_test, predictions))
        mlflow.log_metric("r2_score", r2_score(y_test, predictions))

        mlflow.log_artifact("feature_importance.png")
        mlflow.log_artifact("residual_plot.png")
        mlflow.log_artifact("actual_vs_predicted.png")

        mlflow.sklearn.log_model(rf, artifact_path="model")


print("Training & Logging Selesai!")