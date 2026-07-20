# Credit Card Default Prediction Application

Link - https://kartavya219-credit-card-default-predictor-app-tpmzts.streamlit.app/

An end-to-end production-quality machine learning web application built using **Python**, **scikit-learn**, and **Streamlit** to predict whether a credit card customer will default on their payment next month.

The model is trained on the UCI Credit Card dataset. It utilizes exactly the same preprocessing pipeline, feature engineering, scaling, and training logic as implemented in the original research notebook.

---

## Folder Structure

```
CreditDefaultPrediction/
│
├── app.py                  # Streamlit web application
├── train_model.py          # Model training and artifact serialization script
├── model.pkl               # Serialized Logistic Regression model
├── scaler.pkl              # Serialized StandardScaler object
├── model_features.pkl      # List of features used by the model
├── requirements.txt        # Python package dependencies
├── README.md               # Setup and usage guide
├── UCI_Credit_Card.csv     # The dataset file
└── assets/                 # Folder for styling/assets (optional)
```

---

## Installation & Setup

Ensure you have Python 3.8+ installed. Follow these steps to set up and run the application from a fresh clone:

1. **Clone or navigate to the directory**:
   ```bash
   cd CreditDefaultPrediction
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Train

To train the machine learning model from scratch and save the preprocessing and model artifacts (`model.pkl`, `scaler.pkl`, `model_features.pkl`), run:

```bash
python train_model.py
```

This will:
- Load the dataset `UCI_Credit_Card.csv`
- Perform data preprocessing & feature engineering
- Train the Logistic Regression model using stratified splitting and class weighting
- Evaluate performance metrics (Accuracy, ROC-AUC score, classification report)
- Save the serialized model and standard scaler objects.

---

## How to Launch

Once the model has been trained, launch the Streamlit web application:

```bash
streamlit run app.py
```

This will launch a local server and automatically open the application in your default web browser.

---

## Requirements

The project uses the following package versions:
- `streamlit`
- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
