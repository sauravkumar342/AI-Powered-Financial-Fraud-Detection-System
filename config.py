import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "paysim.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed", "processed_data.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def create_folders():
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)

    os.makedirs(MODEL_DIR, exist_ok=True)

    os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrix"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "roc_curves"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "pr_curves"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "feature_importance"), exist_ok=True)