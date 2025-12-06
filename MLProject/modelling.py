import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Tracking lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("gempa-rf-experiment")

mlflow.end_run()

# Load dataset
df = pd.read_csv("katalog-gempa_preprocessing.csv")

X = df.drop(columns=["mag"])
y = df["mag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()

with mlflow.start_run(nested=True):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, artifact_path="model")
