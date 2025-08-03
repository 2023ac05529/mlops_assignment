# src/model_trainer.py
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import logging

# Set up a logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MLflow Configuration ---
# Use a local file-based database for tracking.
MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
EXPERIMENT_NAME = "Iris_Classifier_Pipeline"
mlflow.set_experiment(EXPERIMENT_NAME)


def run_training():
    """Loads data, trains two models, and logs them to MLflow."""
    logger.info("Starting model training process...")
    
    # Load data from the DVC-tracked file
    # Note: In a real pipeline, you might use dvc.api.open to read this
    df = pd.read_csv('data/iris.csv')
    
    df.columns = ['id', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    df = df.drop('id', axis=1)

    # Simple preprocessing: convert species to numeric
    species_map = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
    df['species'] = df['species'].map(species_map)

    X = df.drop('species', axis=1)
    y = df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2025)
    input_example = X_train.head(5)

    # --- Experiment 1: Logistic Regression ---
    with mlflow.start_run(run_name="Logistic_Regression_Model") as run:
        logger.info("Training Logistic Regression model.")
        
        # Define and train the model with unique parameters
        C_param = 1.2
        log_reg_model = LogisticRegression(C=C_param, max_iter=250, solver='lbfgs')
        log_reg_model.fit(X_train, y_train)

        # Evaluate and log
        predictions = log_reg_model.predict(X_test)
        f1 = f1_score(y_test, predictions, average='weighted')
        
        mlflow.log_param("regularization_strength_C", C_param)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.sklearn.log_model(sk_model=log_reg_model,name="logistic_regression_model",input_example=input_example)
        
        logger.info(f"Logistic Regression F1 Score: {f1:.4f}")

    # --- Experiment 2: Random Forest ---
    with mlflow.start_run(run_name="Random_Forest_Model") as run:
        logger.info("Training Random Forest model.")
        
        # Define and train with unique parameters
        n_estimators_val = 150
        rand_forest_model = RandomForestClassifier(n_estimators=n_estimators_val, max_depth=8, random_state=2025)
        rand_forest_model.fit(X_train, y_train)

        # Evaluate and log
        predictions = rand_forest_model.predict(X_test)
        f1 = f1_score(y_test, predictions, average='weighted')

        mlflow.log_param("num_estimators", n_estimators_val)
        mlflow.log_param("max_depth", 8)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.sklearn.log_model(sk_model=rand_forest_model,name="random_forest_model",input_example=input_example)
        
        logger.info(f"Random Forest F1 Score: {f1:.4f}")

if __name__ == "__main__":
    run_training()