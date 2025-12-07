import mlflow
import mlflow.sklearn
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("gempa-experiment")

base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "katalog-gempa_preprocessing.csv")

df = pd.read_csv(csv_path)

# Pisahkan fitur dan target
X = df.drop(columns=["mag"])
y = df["mag"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()

with mlflow.start_run():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, "model")

    print("MAE:", mae)
