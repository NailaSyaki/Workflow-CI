import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
args = parser.parse_args()

df = pd.read_csv(args.data_path)

X = df.drop(columns=["mag"])
y = df["mag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    
    mlflow.sklearn.log_model(model, "model")
