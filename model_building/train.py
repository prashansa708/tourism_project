# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline
# for model training
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
# for model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# for model serialization
import joblib
# for mlflow tracking
import mlflow
# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
# for JSON operations (for README.md)
import json

# --- MLflow Setup ---
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Tourism_Package_Prediction_Experiment")

# --- Hugging Face API Initialization ---
api = HfApi(token=os.getenv("HF_TOKEN"))

# --- Local Data Directory ---
PROCESSED_DATA_DIR = "tourism_project/data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# --- Download Data from Hugging Face ---
dataset_repo_id = "P-Mishra/tourism_project"

# Files to download
files_to_download = ['X_train.csv', 'X_test.csv', 'y_train.csv', 'y_test.csv']

print("Downloading processed data from Hugging Face...")
for file_name in files_to_download:
    local_file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    try:
        api.hf_hub_download(
            repo_id=dataset_repo_id,
            repo_type="dataset",
            filename=os.path.join("data", "processed", file_name),
            local_dir=PROCESSED_DATA_DIR,
            local_dir_use_symlinks=False
        )
        print(f"Downloaded {file_name} to {local_file_path}")
    except Exception as e:
        print(f"Error downloading {file_name}: {e}")
        exit() # Exit if data download fails

# --- Load Processed Data ---
X_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'))
X_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv')).squeeze()
y_test = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv')).squeeze()

print("Processed data loaded successfully.")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# --- Calculate scale_pos_weight ---
neg, pos = y_train.value_counts()
total = neg + pos
scale_pos_weight = neg / pos
print(f"Class distribution: Negative={neg}, Positive={pos}, Ratio (Negative/Positive)={scale_pos_weight:.2f}")

# --- Define Models and Hyperparameter Grids ---
models_and_params = {
    "DecisionTree": {
        "model": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        "params": {
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    "Bagging": {
        "model": BaggingClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42, class_weight='balanced')),
        "params": {
            'n_estimators': [50, 100, 150],
            'max_samples': [0.7, 0.8, 0.9]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42, class_weight='balanced'),
        "params": {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(random_state=42, estimator=DecisionTreeClassifier(random_state=42, max_depth=1)), # Base estimator often simple for AdaBoost
        "params": {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 1.0]
        }
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        }
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight),
        "params": {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
    }
}

# --- MLflow Experimentation Loop ---
overall_best_recall = -1
overall_best_model = None
overall_best_model_name = ""
overall_best_metrics = {}
overall_best_hyperparams = {}

with mlflow.start_run(run_name="Overall Model Training") as main_run:
    for model_name, config in models_and_params.items():
        with mlflow.start_run(nested=True, run_name=f"Training {model_name}") as nested_run:
            print(f"\n--- Training {model_name} ---")
            mlflow.log_param("model_type", model_name)

            # Create a dummy preprocessor as data is already processed.
            # This ensures the pipeline structure is maintained if needed for deployment consistency.
            preprocessor = Pipeline(steps=[('passthrough', 'passthrough')])
            
            # Create the pipeline with dummy preprocessor and the model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), (model_name.lower(), config["model"])])
            
            # Adjust param_grid keys for the pipeline
            grid_params = {f'{model_name.lower()}__{k}': v for k, v in config["params"].items()}

            grid_search = GridSearchCV(pipeline, grid_params, cv=5, scoring='recall', n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            # Log best parameters
            mlflow.log_params(grid_search.best_params_)

            # Make predictions and evaluate
            y_pred_test = best_model.predict(X_test)
            y_pred_train = best_model.predict(X_train)

            test_accuracy = accuracy_score(y_test, y_pred_test)
            test_precision = precision_score(y_test, y_pred_test, zero_division=0)
            test_recall = recall_score(y_test, y_pred_test, zero_division=0)
            test_f1 = f1_score(y_test, y_pred_test, zero_division=0)

            train_accuracy = accuracy_score(y_train, y_pred_train)
            train_precision = precision_score(y_train, y_pred_train, zero_division=0)
            train_recall = recall_score(y_train, y_pred_train, zero_division=0)
            train_f1 = f1_score(y_train, y_pred_train, zero_division=0)

            metrics = {
                "train_accuracy": train_accuracy,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1_score": train_f1,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1_score": test_f1
            }
            mlflow.log_metrics(metrics)
            print(f"Test Recall: {test_recall:.4f}")

            # Save model locally and as artifact
            models_dir = "models"
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f"{model_name.lower()}_best_model.joblib")
            joblib.dump(best_model, model_path)
            mlflow.log_artifact(model_path, artifact_path="model")
            print(f"Saved {model_name} model to {model_path} and logged as MLflow artifact.")

            # Track the overall best model based on test recall
            if test_recall > overall_best_recall:
                overall_best_recall = test_recall
                overall_best_model = best_model
                overall_best_model_name = model_name
                overall_best_metrics = metrics
                overall_best_hyperparams = grid_search.best_params_

    # Log the overall best model's info in the main run
    if overall_best_model is not None:
        mlflow.log_param("overall_best_model_name", overall_best_model_name)
        mlflow.log_metrics({f"overall_best_{k}": v for k, v in overall_best_metrics.items()})
        # Convert best_hyperparams values to string for logging as parameters if they are complex types
        mlflow.log_params({f"overall_best_hyperparams_{k}": str(v) for k, v in overall_best_hyperparams.items()})
        print(f"\nOverall best model selected: {overall_best_model_name} with Test Recall: {overall_best_recall:.4f}")

        # --- Save Overall Best Model Locally ---
        final_model_path = "best_tourism_package_model.joblib"
        joblib.dump(overall_best_model, final_model_path)
        print(f"Overall best model saved locally as {final_model_path}")
        mlflow.log_artifact(final_model_path, artifact_path="overall_best_model")

        # --- Generate README.md (Model Card) ---
        readme_content = f"""# Tourism Package Prediction Model\n\n## Model Overview\nThis repository contains the best performing machine learning model for predicting customer purchases of the Wellness Tourism Package.\n\n## Best Model Details\n- **Model Type:** {overall_best_model_name}\n- **Best Hyperparameters:**\n```json\n{json.dumps(overall_best_hyperparams, indent=4)}\n```\n\n## Performance Metrics (Test Set)\n- **Accuracy:** {overall_best_metrics['test_accuracy']:.4f}\n- **Precision:** {overall_best_metrics['test_precision']:.4f}\n- **Recall:** {overall_best_metrics['test_recall']:.4f}\n- **F1-Score:** {overall_best_metrics['test_f1_score']:.4f}\n\n## Features Used\nThe model was trained on preprocessed customer and interaction data, including (but not limited to) features such as Age, TypeofContact, CityTier, Occupation, Gender, NumberOfPersonVisiting, PreferredPropertyStar, MaritalStatus, NumberOfTrips, Passport, OwnCar, NumberOfChildrenVisiting, Designation, MonthlyIncome, PitchSatisfactionScore, ProductPitched, NumberOfFollowups, and DurationOfPitch.\n\n## Usage\nTo use this model for inference:\n\n```python\nimport joblib\nimport pandas as pd\n
# Load the model
model = joblib.load('best_tourism_package_model.joblib')

# Prepare your input data (ensure it matches the training data format)\n# Example (replace with actual data structure after preprocessing):\n# new_data = pd.DataFrame(...)\n
# Make a prediction\n# prediction = model.predict(new_data)\n# print(prediction)\n```\n"""
        with open("README.md", "w") as f:
            f.write(readme_content)
        print("Generated README.md for the best model.")

        # --- Hugging Face Model Upload ---
        model_repo_id = "P-Mishra/tourism_package_prediction"
        repo_type = "model"

        try:
            api.repo_info(repo_id=model_repo_id, repo_type=repo_type)
            print(f"Hugging Face model space '{model_repo_id}' already exists. Using it.")
        except RepositoryNotFoundError:
            print(f"Hugging Face model space '{model_repo_id}' not found. Creating new space...")
            create_repo(repo_id=model_repo_id, repo_type=repo_type, private=False)
            print(f"Hugging Face model space '{model_repo_id}' created.")

        print("Uploading best model and README.md to Hugging Face...")
        api.upload_file(
            path_or_fileobj=final_model_path,
            path_in_repo=final_model_path,
            repo_id=model_repo_id,
            repo_type=repo_type,
        )
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=model_repo_id,
            repo_type=repo_type,
        )
        print("Successfully uploaded best model and README.md to Hugging Face.")

    else:
        print("No models were successfully trained or evaluated.")

print("MLflow training completed.")
