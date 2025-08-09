# src/evaluate.py
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_and_promote():
    """
    Loads the latest trained model from a specific experiment and the current
    production model. Compares their performance and promotes the new model if it's better.
    """
    MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
    EXPERIMENT_NAME = "Iris_Classifier_Chennai_2025"
    REGISTERED_MODEL_NAME = "iris-classifier-chennai"
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load the test data
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = pd.read_csv('data/iris.csv', header=None, names=column_names)
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['species'] = df['species'].map(species_map)
    _, X_test, _, y_test = train_test_split(df.drop('species', axis=1), df['species'], test_size=0.2, random_state=42)

    # --- 1. Get the latest trained model ---
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    latest_run = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    ).iloc[0]
    
    latest_model_uri = f"runs:/{latest_run.run_id}/model"
    latest_model = mlflow.pyfunc.load_model(latest_model_uri)
    latest_accuracy = accuracy_score(y_test, latest_model.predict(X_test))
    print(f"Latest trained model accuracy: {latest_accuracy:.4f}")

    # --- 2. Get the current production model ---
    client = mlflow.tracking.MlflowClient()
    try:
        prod_model_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Production"])[0]
        prod_model_uri = prod_model_version.source
        prod_model = mlflow.pyfunc.load_model(prod_model_uri)
        prod_accuracy = accuracy_score(y_test, prod_model.predict(X_test))
        print(f"Current production model (Version {prod_model_version.version}) accuracy: {prod_accuracy:.4f}")
    except IndexError:
        # No production model exists yet
        prod_accuracy = 0
        print("No production model found.")

    # --- 3. Compare and promote if better ---
    if latest_accuracy > prod_accuracy:
        print(f"New model is better. Promoting to Production.")
        # Register the new model
        result = mlflow.register_model(latest_model_uri, REGISTERED_MODEL_NAME)
        # Transition the new version to Production
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=result.version,
            stage="Production",
            archive_existing_versions=True # Move the old Production model to Archived
        )
    else:
        print("Current production model is better or equal. No changes made.")

if __name__ == "__main__":
    evaluate_and_promote()