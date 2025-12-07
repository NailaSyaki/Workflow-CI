import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load dataset gempa
df = pd.read_csv("katalog-gempa_preprocessing.csv")

# Pisahkan fitur & target (mag = magnitudo gempa)
X = df.drop(columns=["mag"])
y = df["mag"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.autolog()

with mlflow.start_run():
    # Model regresi
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )

    # Training
    model.fit(X_train, y_train)

    # Evaluasi
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)

    mlflow.sklearn.log_model(model, artifact_path="model")

    print("=== EVALUASI MODEL GEMPAMU ===")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
