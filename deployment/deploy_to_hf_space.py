import os
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# --- Hugging Face Configuration ---
# Make sure HF_TOKEN environment variable is set
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it for Hugging Face authentication.")

api = HfApi(token=HF_TOKEN)

# Define the Hugging Face Space where the app will be deployed
SPACE_REPO_ID = "P-Mishra/tourism-package-predictor-space" # Replace with your desired Space ID
SPACE_REPO_TYPE = "space"
SPACE_SDK = "docker" # Changed SDK to 'docker' as per error message

# Define the Model details to download
MODEL_REPO_ID = "P-Mishra/tourism_package_prediction"
MODEL_FILE_NAME = "best_tourism_package_model.joblib"

# Local directory for deployment files
default_deployment_dir = "tourism_project/deployment"

# --- 1. Check/Create Hugging Face Space ---
try:
    api.repo_info(repo_id=SPACE_REPO_ID, repo_type=SPACE_REPO_TYPE)
    print(f"Hugging Face Space '{SPACE_REPO_ID}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Hugging Face Space '{SPACE_REPO_ID}' not found. Creating new space...")
    create_repo(repo_id=SPACE_REPO_ID, repo_type=SPACE_REPO_TYPE, private=False, space_sdk=SPACE_SDK)
    print(f"Hugging Face Space '{SPACE_REPO_ID}' created.")
except Exception as e:
    print(f"Error checking or creating space: {e}")
    exit()

# --- 2. Upload Deployment Files ---
local_deployment_files = [
    os.path.join(default_deployment_dir, "app.py"),
    os.path.join(default_deployment_dir, "Dockerfile"),
    os.path.join(default_deployment_dir, "requirements.txt"),
]

print("Uploading deployment files to Hugging Face Space...")
for file_path in local_deployment_files:
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}. Skipping upload.")
        continue
    try:
        # path_in_repo should be just the file name for the root of the space
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=SPACE_REPO_ID,
            repo_type=SPACE_REPO_TYPE,
        )
        print(f"Uploaded {os.path.basename(file_path)} to {SPACE_REPO_ID}")
    except Exception as e:
        print(f"Error uploading {os.path.basename(file_path)}: {e}")

# --- 3. Download the Trained Model ---
local_model_path = os.path.join(default_deployment_dir, MODEL_FILE_NAME)
print(f"Downloading model from {MODEL_REPO_ID}/{MODEL_FILE_NAME}...")
try:
    hf_hub_download(
        repo_id=MODEL_REPO_ID,
        filename=MODEL_FILE_NAME,
        repo_type="model",
        local_dir=default_deployment_dir, # Download directly to the deployment directory
        local_dir_use_symlinks=False
    )
    print(f"Model downloaded to {local_model_path}")
except Exception as e:
    print(f"Error downloading model: {e}")
    exit()

# --- 4. Upload the Downloaded Model to the Space ---
print(f"Uploading downloaded model {MODEL_FILE_NAME} to Hugging Face Space...")
try:
    api.upload_file(
        path_or_fileobj=local_model_path,
        path_in_repo=MODEL_FILE_NAME, # Upload to the root of the space
        repo_id=SPACE_REPO_ID,
        repo_type=SPACE_REPO_TYPE,
    )
    print(f"Successfully uploaded {MODEL_FILE_NAME} to {SPACE_REPO_ID}")
except Exception as e:
    print(f"Error uploading model to space: {e}")

print("Deployment script finished.")
