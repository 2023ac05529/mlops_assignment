# mlflow-server/Dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install MLflow
RUN pip install mlflow scikit-learn

WORKDIR /mlruns

# Expose the port
EXPOSE 5001

# The command to run the MLflow server.
# It will store the database and artifacts in the /mlruns directory,
# which will be our shared volume.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5001", \
     "--backend-store-uri", "sqlite:////mlruns/server.db"]