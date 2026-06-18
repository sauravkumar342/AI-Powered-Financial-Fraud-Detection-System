import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from config import PROCESSED_DATA_PATH, MODEL_DIR


def train_models():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            random_state=42
        ),

        "logistic_regression": LogisticRegression(max_iter=1000),

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss'
        )
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")
        model.fit(X_train, y_train)

        joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")

        trained_models[name] = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test
        }

    return trained_models