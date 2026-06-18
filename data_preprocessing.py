import pandas as pd
import numpy as np
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)

    print("🔍 Checking missing values...")
    print(df.isnull().sum())

    # Drop useless columns
    df = df.drop(["nameOrig", "nameDest"], axis=1)

    # Encode type
    df["type"] = df["type"].map({
        "CASH_OUT": 0,
        "DEBIT": 1,
        "PAYMENT": 2,
        "TRANSFER": 3
    })

    # 🔥 Feature Engineering
    df["balance_diff_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    df["is_zero_orig"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["is_zero_dest"] = (df["oldbalanceDest"] == 0).astype(int)

    df["is_large_txn"] = (df["amount"] > 200000).astype(int)

    # 🚨 IMPORTANT FIXES

    # Replace infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill missing values
    df.fillna(0, inplace=True)

    print("✅ After cleaning NaN:")
    print(df.isnull().sum())

    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✅ Data Preprocessed Successfully")
    return df