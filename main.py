from config import create_folders
from src.data_preprocessing import preprocess_data
from src.advanced_graph import advanced_graph_analysis
from src.model_training import train_models
from src.evaluation import evaluate
from src.graph_fraud import graph_fraud_detection
from src.visualization import (
    plot_confusion,
    plot_roc_all,
    plot_pr_all,
    plot_feature_importance
)
from src.deep_learning import train_deep_learning_models


def main():
    print("🔧 Setting up folders...")
    create_folders()

    print("📊 Preprocessing...")
    df = preprocess_data()

    print("🤖 Training ML models...")
    trained_models = train_models()

    X = df.drop("isFraud", axis=1)

    results_dict = {}

    for name, data in trained_models.items():
        model = data["model"]
        X_test = data["X_test"]
        y_test = data["y_test"]

        print(f"\n📈 Evaluating {name}...")

        results = evaluate(model, X_test, y_test)

        print("Accuracy:", results["accuracy"])
        print("AUC Score:", results["auc"])
        print(results["report"])

        plot_confusion(results["confusion_matrix"], name)
        plot_feature_importance(model, X.columns, name)

        results_dict[name] = results

    plot_roc_all(results_dict)
    plot_pr_all(results_dict)
    print("\n🔗 Running Graph-Based Fraud Detection...")
    graph_fraud_detection()

    # 🧠 ADVANCED GRAPH AI
    print("\n🧠 Running Advanced Graph AI...")
    advanced_graph_analysis()

    # ======================================
    # 🔥 DEEP LEARNING
    # ======================================
    print("\n🧠 Training Deep Learning Models...")

    dl_results = train_deep_learning_models()

    for name, res in dl_results.items():
        print(f"\n{name} Results:")
        print("Accuracy:", res["accuracy"])
        print("AUC:", res["auc"])


if __name__ == "__main__":
    main()