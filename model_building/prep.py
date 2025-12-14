# for data manipulation
import pandas as pd
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # LabelEncoder is not directly used in the final version
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/P-Mishra/tourism_project/tourism.csv" # Corrected path for raw data
PROCESSED_DATA_DIR = "tourism_project/data/processed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Data Cleaning
df.drop(columns=['Unnamed: 0', 'CustomerID'], inplace=True)
print("Dropped 'Unnamed: 0' and 'CustomerID' columns.")

# Separate target variable
y = df['ProdTaken']
X = df.drop(columns=['ProdTaken'])
print("Separated target variable 'ProdTaken' from features.")

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include='object').columns
numerical_features = X.select_dtypes(exclude='object').columns
print(f"Identified categorical features: {list(categorical_features)}")
print(f"Identified numerical features: {list(numerical_features)}")

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
print("One-hot encoded categorical features.")

# Scale numerical features
scaler = StandardScaler()
X_numerical_scaled = pd.DataFrame(scaler.fit_transform(X[numerical_features]), columns=numerical_features, index=X.index)
print("Scaled numerical features using StandardScaler.")

# Concatenate processed features
X_processed = pd.concat([X_numerical_scaled, X_encoded.drop(columns=numerical_features)], axis=1)
print(f"Concatenated processed features. Shape: {X_processed.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print("Split data into training and testing sets.")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

# Save processed data
X_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(PROCESSED_DATA_DIR, 'y_test.csv'), index=False)
print(f"Processed data saved to {PROCESSED_DATA_DIR}")


files = ["X_train.csv","X_test.csv","y_train.csv","y_test.csv"]


for file_name in files:
    local_file_path = os.path.join(PROCESSED_DATA_DIR, file_name)
    api.upload_file(
        path_or_fileobj=local_file_path,
        path_in_repo=os.path.join("data", "processed", file_name),  # Upload to data/processed in the repo
        repo_id="P-Mishra/tourism_project", # Corrected repo_id
        repo_type="dataset",
    )
