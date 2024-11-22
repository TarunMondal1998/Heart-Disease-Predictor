import pickle
import streamlit as st
import numpy as np

# Importing the model and dataset
xgb = pickle.load(open('xgb.pkl', 'rb'))
data = pickle.load(open('heart_data.pkl', 'rb'))

# Page configuration
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #edbeca;
        font-family: Arial, sans-serif;
    }
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        color: #333333;
    }
    .sub-text {
        text-align: center;
        color: #555555;
    }
    .banner {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 20px 0;
    }
    .banner-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: red;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application Title
st.markdown('<div class="main-title">Heart Disease Predictor ❤️</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Empowered by Machine Learning</div>', unsafe_allow_html=True)

# Add a banner with an image and text
st.markdown(
    """
    <div class="banner">
        <div class="banner-text">Heart Health Matters</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("### Please provide the following details for the prediction:")

# Input fields organized in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=1)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", options=[1, 0])
    cp = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (in mm Hg)", min_value=50, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (in mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["True", "False"])

with col2:
    restecg = st.selectbox("Resting ECG Results", options=["Normal", "ST-T Abnormality", "LV Hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise-Induced Angina", options=["Yes", "No"])
    oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", options=["Upsloping", "Flat", "Downsloping"])
    ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0, step=1)
    thal = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])

# Prediction button
if st.button("Predict"):
    # Map input data correctly
    cp_dict = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
    fbs_dict = {"True": 1, "False": 0}
    restecg_dict = {"Normal": 0, "ST-T Abnormality": 1, "LV Hypertrophy": 2}
    exang_dict = {"Yes": 1, "No": 0}
    slope_dict = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
    thal_dict = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}

    input_data = np.array([[
        age,
        sex,
        cp_dict[cp],
        trestbps,
        chol,
        fbs_dict[fbs],
        restecg_dict[restecg],
        thalach,
        exang_dict[exang],
        oldpeak,
        slope_dict[slope],
        ca,
        thal_dict[thal]
    ]])

    # Prediction
    prediction = xgb.predict(input_data)[0]

    # Display results
    st.markdown("### Prediction Results:")
    if prediction == 1:
        st.success("The System predicts that the person has **Heart Disease**.")
    else:
        st.success("The System predicts that the person does **not have Heart Disease**.")

