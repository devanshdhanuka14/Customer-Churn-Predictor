import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("churn_model.pkl")

# Fixed decision threshold
THRESHOLD = 0.40

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")

st.title("📊 Customer Churn Prediction System")
st.write("Predict the probability of customer churn using a trained ML model.")
st.caption("Model: Logistic Regression | Threshold: 0.40 | Recall Optimized")

st.markdown("---")

# -------------------------
# User Inputs
# -------------------------

st.subheader("Customer Information")

tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0)

senior = st.selectbox("Senior Citizen", [0, 1])

gender = st.selectbox("Gender", ["Male", "Female"])

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

phone_service = st.selectbox("Phone Service", ["Yes", "No"])

multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])

internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])

online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])

streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

st.markdown("---")

# -------------------------
# Prediction
# -------------------------

if st.button("Predict Churn Risk"):

    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "SeniorCitizen": [senior],
        "gender": [gender],
        "Partner": [partner],
        "Dependents": [dependents],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method]
    })

    # Get probability
    prob = model.predict_proba(input_data)[:, 1][0]

    prediction = int(prob >= THRESHOLD)

    st.subheader(f"Churn Probability: {prob:.2%}")

    if prediction == 1:
        st.error("⚠️ Prediction: Likely to Churn")
    else:
        st.success("✅ Prediction: Not Likely to Churn")

    # Risk Category
    st.markdown("### Risk Category")

    if prob < 0.30:
        st.success("🟢 LOW RISK")
    elif prob < 0.60:
        st.warning("🟡 MEDIUM RISK")
    else:
        st.error("🔴 HIGH RISK")



st.markdown("---")
st.markdown("Built by Devansh Dhanuka")