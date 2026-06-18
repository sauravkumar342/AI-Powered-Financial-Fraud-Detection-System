import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd

from config import MODEL_DIR


def load_model(model_name="xgboost"):
    return joblib.load(f"{MODEL_DIR}/{model_name}.pkl")


def preprocess_input(data):
    type_mapping = {
        "CASH_OUT": 0,
        "DEBIT": 1,
        "PAYMENT": 2,
        "TRANSFER": 3
    }

    data["type"] = type_mapping.get(data["type"], 2)

    # 🔥 SAME FEATURES AS TRAINING
    data["balance_diff_orig"] = data["oldbalanceOrg"] - data["newbalanceOrig"]
    data["balance_diff_dest"] = data["newbalanceDest"] - data["oldbalanceDest"]

    data["is_zero_orig"] = int(data["oldbalanceOrg"] == 0)
    data["is_zero_dest"] = int(data["oldbalanceDest"] == 0)

    data["is_large_txn"] = int(data["amount"] > 200000)

    return pd.DataFrame([data])


def get_user_input():
    print("\n🔍 Enter Transaction Details:\n")

    return {
        "step": int(input("Step: ")),
        "type": input("Type: ").upper(),
        "amount": float(input("Amount: ")),
        "oldbalanceOrg": float(input("Old Sender Balance: ")),
        "newbalanceOrig": float(input("New Sender Balance: ")),
        "oldbalanceDest": float(input("Old Receiver Balance: ")),
        "newbalanceDest": float(input("New Receiver Balance: ")),
        "isFlaggedFraud": int(input("Flagged Fraud (0/1): "))
    }


def predict():
    model = load_model()

    user_data = get_user_input()
    processed = preprocess_input(user_data)

    pred = model.predict(processed)[0]
    prob = model.predict_proba(processed)[0][1]

    print("\n🔎 RESULT:")
    if prob > 0.8:
        print(f"🚨 HIGH RISK FRAUD (Risk: {prob:.2f})")
    elif prob > 0.5:
        print(f"⚠️ SUSPICIOUS TRANSACTION (Risk: {prob:.2f})")
    else:
        print(f"✅ SAFE TRANSACTION (Risk: {prob:.2f})")

if __name__ == "__main__":
    predict()