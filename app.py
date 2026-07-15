import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration with a custom title and layout
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)


# Load cached artifacts
@st.cache_resource
def load_model_artifacts():
    """Load model, scaler, and features metadata."""
    model_path = "model.pkl"
    scaler_path = "scaler.pkl"
    
    # Handle relative paths from folder structure context
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

try:
    model, scaler = load_model_artifacts()
except Exception as e:
    st.error(f"Error loading model assets. Please ensure `train_model.py` has been run. Details: {e}")
    st.stop()

def preprocess_input(raw_inputs):
    """
    Applies the exact same feature engineering, ordering, and scaling pipeline
    as the model training phase.
    """
    # Map categoricals to numeric representations
    sex = 1 if raw_inputs['SEX'] == "Male" else 2
    
    edu_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
    education = edu_map[raw_inputs['EDUCATION']]
    
    marriage_map = {"Married": 1, "Single": 2, "Others": 3}
    marriage = marriage_map[raw_inputs['MARRIAGE']]
    
    age = raw_inputs['AGE']
    limit_bal = raw_inputs['LIMIT_BAL']
    
    bill_amts = [
        raw_inputs['BILL_AMT1'], raw_inputs['BILL_AMT2'], raw_inputs['BILL_AMT3'],
        raw_inputs['BILL_AMT4'], raw_inputs['BILL_AMT5'], raw_inputs['BILL_AMT6']
    ]
    
    pay_amts = [
        raw_inputs['PAY_AMT1'], raw_inputs['PAY_AMT2'], raw_inputs['PAY_AMT3'],
        raw_inputs['PAY_AMT4'], raw_inputs['PAY_AMT5'], raw_inputs['PAY_AMT6']
    ]
    
    pay_statuses = [
        raw_inputs['PAY_0'], raw_inputs['PAY_2'], raw_inputs['PAY_3'],
        raw_inputs['PAY_4'], raw_inputs['PAY_5'], raw_inputs['PAY_6']
    ]
    
    # Calculate engineered features
    avg_bill_amt = np.mean(bill_amts)
    avg_pay_amt = np.mean(pay_amts)
    
    late_payment_count = sum(1 for status in pay_statuses if status > 0)
    max_delay = max(pay_statuses)
    
    # Apply log scaling & clipping
    log_limit_bal = np.log1p(limit_bal)
    log_bill_amt_avg = np.log1p(max(0.0, avg_bill_amt))
    log_pay_amt_avg = np.log1p(max(0.0, avg_pay_amt))
    
    # Assemble feature vector in the precise training column order
    features_dict = {
        'SEX': [sex],
        'EDUCATION': [education],
        'MARRIAGE': [marriage],
        'AGE': [age],
        'log_limit_bal': [log_limit_bal],
        'log_bill_amt_avg': [log_bill_amt_avg],
        'log_pay_amt_avg': [log_pay_amt_avg],
        'late_payment_count': [late_payment_count],
        'max_delay': [max_delay]
    }
    
    df_features = pd.DataFrame(features_dict)
    
    # Standard scaling
    scaled_features = scaler.transform(df_features)
    return scaled_features

def predict_default(processed_features):
    """Predict class labels and probabilities."""
    prediction = model.predict(processed_features)[0]
    probability = model.predict_proba(processed_features)[0, 1]
    return prediction, probability

# ----------------- UI Rendering -----------------

st.markdown("<h1 style='text-align: center;'>Credit Card Default Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Customer Information</h3><br>", unsafe_allow_html=True)


# Main form layout
with st.container():
    st.markdown("<h3 style='text-align: center;'>Demographic & Account Profile</h3><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        limit_bal = st.number_input(
            "Credit Limit", 
            min_value=0, 
            value=50000, 
            step=1000, 
            help="Total credit limit of the customer (cannot be negative)"
        )
        age = st.number_input(
            "Age", 
            min_value=0, 
            max_value=120, 
            value=30, 
            step=1, 
            help="Age in years (cannot be negative)"
        )
        
    with col2:
        sex = st.selectbox(
            "Gender", 
            options=["Male", "Female"],
            help="Gender classification of the customer"
        )
        education = st.selectbox(
            "Education Level", 
            options=["Graduate School", "University", "High School", "Others"],
            help="Highest completed education level"
        )
        
    with col3:
        marriage = st.selectbox(
            "Marital Status", 
            options=["Married", "Single", "Others"],
            help="Marital status of the customer"
        )

# Monthly billing metrics tabs/columns to present inputs cleanly
with st.container():
    st.markdown("<br><h3 style='text-align: center;'>Payment History Details</h3><br>", unsafe_allow_html=True)
    
    # We group the 6 monthly columns side by side
    month_cols = st.columns(6)
    months = ["Sept 2005", "Aug 2005", "Jul 2005", "Jun 2005", "May 2005", "Apr 2005"]
    
    # Storage for monthly status, bill amounts, and pay amounts
    pay_statuses = []
    bill_amts = []
    pay_amts = []
    
    for idx, col in enumerate(month_cols):
        with col:
            st.markdown(f"**{months[idx]}**")
            
            # Repayment status input
            # Values: -2 = No consumption, -1 = Paid in full, 0 = Revolving, 1 = Delay 1 month, etc.
            status = st.number_input(
                f"Status",
                min_value=-2,
                max_value=9,
                value=0,
                step=1,
                key=f"pay_status_{idx}",
                help="-2: No consumption, -1: Paid in full, 0: Revolving credit, 1-9: Payment delay in months"
            )
            pay_statuses.append(status)
            
            # Bill amount input
            bill = st.number_input(
                f"Bill Amount",
                value=0.0,
                step=100.0,
                key=f"bill_amt_{idx}",
                help="Statement bill amount (can be zero or negative)"
            )
            bill_amts.append(bill)
            
            # Pay amount input
            pay = st.number_input(
                f"Paid Amount",
                min_value=0.0,
                value=0.0,
                step=100.0,
                key=f"pay_amt_{idx}",
                help="Amount paid in the statement period (cannot be negative)"
            )
            pay_amts.append(pay)

# Map UI list variables back to correct dictionary keys for validation & processing
raw_inputs = {
    'SEX': sex,
    'EDUCATION': education,
    'MARRIAGE': marriage,
    'AGE': age,
    'LIMIT_BAL': limit_bal,
    'PAY_0': pay_statuses[0],
    'PAY_2': pay_statuses[1],
    'PAY_3': pay_statuses[2],
    'PAY_4': pay_statuses[3],
    'PAY_5': pay_statuses[4],
    'PAY_6': pay_statuses[5],
    'BILL_AMT1': bill_amts[0],
    'BILL_AMT2': bill_amts[1],
    'BILL_AMT3': bill_amts[2],
    'BILL_AMT4': bill_amts[3],
    'BILL_AMT5': bill_amts[4],
    'BILL_AMT6': bill_amts[5],
    'PAY_AMT1': pay_amts[0],
    'PAY_AMT2': pay_amts[1],
    'PAY_AMT3': pay_amts[2],
    'PAY_AMT4': pay_amts[3],
    'PAY_AMT5': pay_amts[4],
    'PAY_AMT6': pay_amts[5],
}

st.write("")
predict_clicked = st.button("Predict Default Status")

if predict_clicked:
    # ---------------- Validation Checks ----------------
    validation_passed = True
    error_msg = ""
    
    if age < 0:
        validation_passed = False
        error_msg += "Age cannot be negative.\n"
    if limit_bal < 0:
        validation_passed = False
        error_msg += "Credit Limit cannot be negative.\n"
    for idx, val in enumerate(pay_amts):
        if val < 0:
            validation_passed = False
            error_msg += f"Payment amount {idx+1} cannot be negative.\n"
            
    if not validation_passed:
        st.error(f"Input Validation Failed:\n{error_msg}")
    else:
        # Preprocess & Predict
        scaled_features = preprocess_input(raw_inputs)
        prediction, probability = predict_default(scaled_features)
        
        # Display Prediction results (ONLY the requested information)
        st.write("")
        st.write("---")
        
        pred_label = "Default" if prediction == 1 else "No Default"
        card_class = "risk-card" if prediction == 1 else "safe-card"
        
        st.markdown(f"""
            <div class="prediction-card {card_class}">
                <div style="font-size: 1.2rem; text-transform: uppercase; letter-spacing: 2px;">Prediction</div>
                <div class="result-title">{pred_label}</div>
                <div class="probability-label">Probability of Default</div>
                <div class="probability-val">{probability * 100:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Streamlit Progress bar matching default probability
        st.progress(float(probability))
