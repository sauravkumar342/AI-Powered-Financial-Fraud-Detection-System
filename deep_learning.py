import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from config import PROCESSED_DATA_PATH, MODEL_DIR


def train_deep_learning_models():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    # 🔥 SCALE (VERY IMPORTANT)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    results = {}

    # =====================================
    # 🔥 ANN MODEL
    # =====================================
    print("\n🚀 Training ANN...")

    ann = Sequential([
        Dense(128, activation='relu', input_dim=X_train.shape[1]),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    ann.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    ann.fit(
        X_train, y_train,
        epochs=8,
        batch_size=512,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=2)],
        verbose=1
    )

    y_prob_ann = ann.predict(X_test)
    y_pred_ann = (y_prob_ann > 0.5).astype(int)

    results["ANN"] = {
        "accuracy": accuracy_score(y_test, y_pred_ann),
        "auc": roc_auc_score(y_test, y_prob_ann)
    }

    ann.save(f"{MODEL_DIR}/ann_model.h5")

    # =====================================
    # 🔥 LSTM MODEL (BEST)
    # =====================================
    print("\n🚀 Training LSTM...")

    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    lstm = Sequential([
        LSTM(128, input_shape=(1, X_train.shape[1])),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    lstm.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    lstm.fit(
        X_train_lstm, y_train,
        epochs=8,
        batch_size=512,
        validation_split=0.2,
        callbacks=[EarlyStopping(patience=2)],
        verbose=1
    )

    y_prob_lstm = lstm.predict(X_test_lstm)
    y_pred_lstm = (y_prob_lstm > 0.5).astype(int)

    results["LSTM"] = {
        "accuracy": accuracy_score(y_test, y_pred_lstm),
        "auc": roc_auc_score(y_test, y_prob_lstm)
    }

    lstm.save(f"{MODEL_DIR}/lstm_model.h5")

    return results