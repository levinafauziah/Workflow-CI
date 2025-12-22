import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=115)
    parser.add_argument("--max_depth", type=int, default=22)
    args = parser.parse_args()

    df = pd.read_csv("dataset_preprocessed/data_clean.csv")
    X = df.drop("rent", axis=1)
    y = df["rent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run(run_name="house_rent_ci"):
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mlflow.log_params(vars(args))
        mlflow.log_metric("mse", mean_squared_error(y_test, preds))
        mlflow.log_metric("r2", r2_score(y_test, preds))

        mlflow.sklearn.log_model(model, artifact_path="model")

        plt.figure()
        sns.scatterplot(x=preds, y=y_test - preds)
        plt.axhline(0)
        plt.savefig("residual_plot.png")
        mlflow.log_artifact("residual_plot.png")

        mlflow.sklearn.log_model(
            rf,
            artifact_path="model",
            registered_model_name="house_rent_model"
        )

print("Training & Logging Selesai!")