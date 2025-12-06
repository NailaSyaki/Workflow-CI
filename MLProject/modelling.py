import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Load dataset
df = pd.read_csv("katalog-gempa_preprocessing.csv")

# Tentukan X dan y
X = df.drop(columns=["mag"])
y = df["mag"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Aktifkan autolog MLflow
mlflow.autolog()

# Training model
with mlflow.start_run():

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, artifact_path="model")

    print("Training selesai. Model logged ke artifacts/model")
