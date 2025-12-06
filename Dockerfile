FROM python:3.10-slim
WORKDIR /app
RUN pip install mlflow scikit-learn pandas
COPY model /app/model
EXPOSE 5000
CMD ["mlflow", "models", "serve", "-m", "/app/model", "--host", "0.0.0.0", "--port", "5000", "--no-conda"]
