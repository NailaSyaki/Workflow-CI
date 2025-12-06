import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("gempa-rf-experiment")

# Load dataset (hasil preprocessing)
df = pd.read_csv("katalog-gempa_preprocessing.csv")

# Tentukan X dan y
X = df.drop(columns=["mag"])
y = df["mag"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.end_run()
mlflow.autolog()

with mlflow.start_run():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, artifact_path="model")
    print("Training selesai. Model tersimpan di mlruns/<exp>/<run>/artifacts/model")
