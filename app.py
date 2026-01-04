import streamlit as st
import pandas as pd
import joblib

preprocessor = joblib.load("models/preprocessor.pkl")
model = joblib.load("models/xgb_model.pkl")
threshold = joblib.load("models/threshold.pkl")

st.title("Customer Churn Prediction Demo")

st.write("Enter customer information. Missing values are allowed.")

# Select boxes
Gender = st.selectbox("Gender", ["Male", "Female", None])
Senior = st.selectbox("Senior Citizen", ['Yes', 'No', None])
Senior_val = (
   1 if Senior == 'Yes' else
   0 if Senior == 'No' else
   None)
Partner = st.selectbox("Partner", ["Yes", "No", None])
Dependents = st.selectbox("Dependents", ["Yes", "No", None])
Phone = st.selectbox("Phone Service", ["Yes", "No", None])
Multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service", None])
Internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No", None])
OnlineSecurity = st.selectbox('Online Service', ['Yes', 'No', 'No internet service', None])
OnlineBackup = st.selectbox('Online Backup', ['Yes', 'No', 'No internet service', None])
DeviceProtection = st.selectbox('Device Protection', ['Yes', 'No', 'No internet service', None])
TechSupport = st.selectbox('Tech Support', ['Yes', 'No', 'No internet service', None])
StreamingTV = st.selectbox('Streaming TV', ['Yes', 'No', 'No internet service', None])
StreamingMovies = st.selectbox('Streaming Movies', ['Yes', 'No', 'No internet service', None])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year", None])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No", None])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)", None])
# Number inputs
Monthly_missing = st.checkbox("Monthly Charges missing")
if Monthly_missing:
    Monthly = None
else:
    Monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

Tenure_missing = st.checkbox("Tenure missing")
if Tenure_missing:
    Tenure = None
else:
    Tenure = st.number_input("Tenure (months)", min_value=0, value=5)

Total_missing = st.checkbox("Total Charges missing")
if Total_missing:
    Total = None
else:
    Total = st.number_input("Total Charges", min_value=0.0, value=400.0)


# Prediction button
input_values = [Gender, Senior_val, Partner, Dependents, Tenure, Phone,
                Multiple_lines, Internet, OnlineSecurity, OnlineBackup,
                DeviceProtection, TechSupport, StreamingTV, StreamingMovies,
                Contract, PaperlessBilling, PaymentMethod, Monthly, Total]
missing_count = sum(x is None for x in input_values)

if st.button("Predict Churn"):
    user_input = pd.DataFrame([{
        "gender": Gender,
        "SeniorCitizen": Senior_val,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": Tenure,
        "PhoneService": Phone,
        "MultipleLines": Multiple_lines,
        "InternetService": Internet,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": Monthly,
        "TotalCharges": Total
    }])
    if missing_count > 6:
        st.warning(f'{missing_count} fields are missing. Prediction may be unreliable.')

    X_processed = preprocessor.transform(user_input)
    prob = model.predict_proba(X_processed)[:, 1][0]
    pred = "Yes" if prob >= threshold else "No"

    st.write(f"**Churn Probability:** {prob:.2%}")
    st.write(f"**Predicted Churn:** {pred}")