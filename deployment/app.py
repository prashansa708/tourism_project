# Import necessary libraries
import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import HfApi, hf_hub_download
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Constants for Hugging Face --- 
MODEL_REPO_ID = "P-Mishra/tourism_package_prediction"
MODEL_FILE_NAME = "best_tourism_package_model.joblib"
DATASET_REPO_ID = "P-Mishra/tourism_project"
RAW_DATA_FILE_NAME = "tourism.csv"

# Initialize HfApi (token will be picked from environment variables if set)
api = HfApi()

# --- Cached function to load the model --- 
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILE_NAME,
            repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Cached function to set up preprocessors --- 
@st.cache_resource
def setup_preprocessors():
    try:
        # Download raw data to fit preprocessors
        raw_data_path = hf_hub_download(
            repo_id=DATASET_REPO_ID,
            filename=os.path.join("data", RAW_DATA_FILE_NAME),
            repo_type="dataset"
        )
        raw_df = pd.read_csv(raw_data_path)

        # Drop irrelevant columns from raw data for preprocessing setup
        raw_df.drop(columns=['Unnamed: 0', 'CustomerID', 'ProdTaken'], inplace=True)

        # Identify features (consistent with prep.py)
        numerical_features = raw_df.select_dtypes(exclude='object').columns.tolist()
        categorical_features = raw_df.select_dtypes(include='object').columns.tolist()

        # Fit StandardScaler on numerical features
        scaler = StandardScaler()
        scaler.fit(raw_df[numerical_features])

        # Fit OneHotEncoder on categorical features
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
        encoder.fit(raw_df[categorical_features])

        # Generate encoded feature names
        encoded_categorical_names = encoder.get_feature_names_out(categorical_features)
        all_encoded_feature_names = numerical_features + encoded_categorical_names.tolist()

        return scaler, encoder, numerical_features, categorical_features, all_encoded_feature_names
    except Exception as e:
        st.error(f"Error setting up preprocessors: {e}")
        return None, None, None, None, None

# --- Load model and preprocessors globally --- 
model = load_model()
scaler, encoder, numerical_features, categorical_features, all_encoded_feature_names = setup_preprocessors()


# --- Streamlit UI --- 
st.set_page_config(page_title="Wellness Tourism Package Predictor", layout="wide")
st.title("Wellness Tourism Package Purchase Predictor")
st.write("Predict whether a customer will purchase the Wellness Tourism Package based on their details.")

if model is None or scaler is None or encoder is None:
    st.warning("Model or preprocessors could not be loaded. Please check logs.")
else:
    with st.form("prediction_form"):
        st.header("Customer Details")

        # Input fields for all 16 features
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.slider("Age", 18, 80, 30)
            typeofcontact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])
            citytier = st.selectbox("City Tier", [1, 2, 3])
            durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=120, value=15)
            numberofpersonvisiting = st.slider("Number of Persons Visiting", 1, 10, 1)
            numberoffollowups = st.slider("Number of Follow-ups", 0, 10, 2)

        with col2:
            preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
            numberoftrips = st.slider("Number of Trips Annually", 0, 10, 1)
            passport = st.selectbox("Has Passport?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
            pitchsatisfactions_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
            owncar = st.selectbox("Owns Car?", [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')

        with col3:
            numberofchildrenvisiting = st.slider("Number of Children Visiting", 0, 5, 0)
            monthlyincome = st.number_input("Monthly Income", min_value=0, value=30000, step=1000)
            occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer'])
            gender = st.selectbox("Gender", ['Male', 'Female'])
            productpitched = st.selectbox("Product Pitched", ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King', 'Premium'])
            maritalstatus = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])
            designation = st.selectbox("Designation", ['Executive', 'Senior Manager', 'Manager', 'AVP', 'VP'])

        submitted = st.form_submit_button("Predict Purchase")

        if submitted:
            # Create a DataFrame from inputs
            input_data = pd.DataFrame([{
                'Age': age,
                'TypeofContact': typeofcontact,
                'CityTier': citytier,
                'DurationOfPitch': durationofpitch,
                'NumberOfPersonVisiting': numberofpersonvisiting,
                'NumberOfFollowups': numberoffollowups,
                'PreferredPropertyStar': preferredpropertystar,
                'NumberOfTrips': numberoftrips,
                'Passport': passport,
                'PitchSatisfactionScore': pitchsatisfactions_score,
                'OwnCar': owncar,
                'NumberOfChildrenVisiting': numberofchildrenvisiting,
                'MonthlyIncome': monthlyincome,
                'Occupation': occupation,
                'Gender': gender,
                'ProductPitched': productpitched,
                'MaritalStatus': maritalstatus,
                'Designation': designation
            }])

            # Separate numerical and categorical features for preprocessing
            input_numerical = input_data[numerical_features]
            input_categorical = input_data[categorical_features]

            # Scale numerical features
            scaled_numerical = scaler.transform(input_numerical)
            scaled_numerical_df = pd.DataFrame(scaled_numerical, columns=numerical_features, index=input_data.index)

            # One-hot encode categorical features
            encoded_categorical = encoder.transform(input_categorical)
            encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features), index=input_data.index)

            # Concatenate processed features
            processed_input = pd.concat([scaled_numerical_df, encoded_categorical_df], axis=1)

            # Align columns with training data (fill missing columns with 0)
            final_input_df = pd.DataFrame(columns=all_encoded_feature_names)
            final_input_df = pd.concat([final_input_df, processed_input], ignore_index=True)
            final_input_df = final_input_df.fillna(0)

            # Make prediction
            prediction = model.predict(final_input_df)
            prediction_proba = model.predict_proba(final_input_df)

            st.subheader("Prediction Result:")
            if prediction[0] == 1:
                st.success(f"Customer is LIKELY to purchase the Wellness Tourism Package! (Probability: {prediction_proba[0][1]:.2f})")
            else:
                st.info(f"Customer is NOT likely to purchase the Wellness Tourism Package. (Probability: {prediction_proba[0][0]:.2f})")
