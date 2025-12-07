import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ========== Path Dataset ==========
base_path = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_path, "katalog-gempa_preprocessing.csv")

print("📌 Loading dataset from:", csv_path)
df = pd.read_csv(csv_path)

# ========== Pisahkan fitur dan target ==========
X = df.drop(columns=["mag"])
y = df["mag"]

# ========== Train-test split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tracking_uri = "file:" + os.path.join(base_path, "mlruns")
print("📌 Tracking URI:", tracking_uri)

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("gempa-experiment")

mlflow.autolog()

with mlflow.start_run():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"🎯 MAE: {mae}")
