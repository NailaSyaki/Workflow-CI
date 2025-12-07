import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Setup Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--run_name", type=str, default="Model Random Forest")
args = parser.parse_args()

df = pd.read_csv(args.data_path)
X = df.drop(columns=["mag"])
y = df["mag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Start MLflow run
with mlflow.start_run(run_name=args.run_name):
    mlflow.sklearn.autolog()

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mlflow.log_metric("mae_manual", mae)

    print("MAE:", mae)

    print("Saving model artifact to 'model' directory...")
    mlflow.sklearn.log_model(model, "model")
