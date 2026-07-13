"""
Credit Card Default Prediction - Model Training Script
This script trains a Logistic Regression model to predict the probability of credit card default.
It replicates the exact feature engineering, train-test split, and scaling pipeline from the source notebook.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import joblib

# Paths
DATASET_PATH = "UCI_Credit_Card.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURES_PATH = "model_features.pkl"

def load_data(filepath):
    """Load the UCI credit card dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Apply preprocessing and feature engineering steps matching the source notebook.
    
    WARNING: Do not change the preprocessing logic, column names, or feature order.
    """
    # 1. Rename target column to 'default'
    if 'default.payment.next.month' in df.columns:
        df = df.rename(columns={'default.payment.next.month': 'default'})
    
    # 2. Map education anomalies: replace [0, 5, 6] with 4 (Others)
    df['EDUCATION'] = df['EDUCATION'].replace([0, 5, 6], 4)
    
    # 3. Map marriage anomalies: replace 0 with 3 (Others)
    df['MARRIAGE'] = df['MARRIAGE'].replace(0, 3)
    
    # 4. Feature engineering: average bill amount and average payment amount
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    
    df['avg_bill_amt'] = df[bill_cols].mean(axis=1)
    df['avg_pay_amt'] = df[pay_amt_cols].mean(axis=1)
    
    # 5. Feature engineering: late payment counts and max delay
    pay_status_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    df['late_payment_count'] = (df[pay_status_cols] > 0).sum(axis=1)
    df['max_delay'] = df[pay_status_cols].max(axis=1)
    
    # 6. Log transformation of skewed columns
    df['log_limit_bal'] = np.log1p(df['LIMIT_BAL'])
    df['log_bill_amt_avg'] = np.log1p(df['avg_bill_amt'].clip(lower=0))
    df['log_pay_amt_avg'] = np.log1p(df['avg_pay_amt'].clip(lower=0))
    
    return df

def train_and_evaluate():
    """Train the model and save artifacts."""
    print("Loading dataset...")
    df = load_data(DATASET_PATH)
    
    print("Preprocessing and engineering features...")
    df = preprocess_data(df)
    
    # Feature columns used by the trained model
    feature_cols = [
        'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'log_limit_bal', 'log_bill_amt_avg', 'log_pay_amt_avg',
        'late_payment_count', 'max_delay'
    ]
    
    X = df[feature_cols]
    y = df['default']
    
    # Stratified Train-Test Split (matching the notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scaling
    print("Fitting scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression model
    print("Training Logistic Regression model...")
    log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    # Internal Evaluation
    print("\n=== Model Internal Evaluation ===")
    y_pred = log_reg.predict(X_test_scaled)
    y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Serialize artifacts
    print("\nSaving serialized model components...")
    joblib.dump(log_reg, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(feature_cols, FEATURES_PATH)
    print("Model saved to:", MODEL_PATH)
    print("Scaler saved to:", SCALER_PATH)
    print("Features list saved to:", FEATURES_PATH)
    print("Training completed successfully.")

if __name__ == "__main__":
    train_and_evaluate()
