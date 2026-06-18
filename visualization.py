import matplotlib.pyplot as plt
import numpy as np
from config import OUTPUT_DIR


def plot_confusion(cm, name):
    plt.figure(figsize=(6, 5))

    plt.imshow(cm, cmap='Blues')
    plt.title(f"{name} Confusion Matrix")

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # 🔥 ADD NUMBERS INSIDE MATRIX
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i,
                format(cm[i, j], 'd'),
                ha="center",
                va="center",
                color="black",
                fontsize=12
            )

    plt.colorbar()

    plt.xticks([0, 1])
    plt.yticks([0, 1])

    plt.tight_layout()

    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix/{name}.png")
    plt.close()


def plot_feature_importance(model, feature_names, name):
    if hasattr(model, "feature_importances_"):
        plt.figure(figsize=(8, 6))
        plt.barh(feature_names, model.feature_importances_)
        plt.title(f"{name} Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/feature_importance/{name}.png")
        plt.close()


def plot_roc_all(results_dict):
    plt.figure()

    for name, res in results_dict.items():
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(res["y_test"], res["y_prob"])
        plt.plot(fpr, tpr, label=name)

    plt.legend()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.savefig(f"{OUTPUT_DIR}/roc_curves/all_models.png")
    plt.close()


def plot_pr_all(results_dict):
    plt.figure()

    for name, res in results_dict.items():
        from sklearn.metrics import precision_recall_curve

        precision, recall, _ = precision_recall_curve(res["y_test"], res["y_prob"])
        plt.plot(recall, precision, label=name)

    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.savefig(f"{OUTPUT_DIR}/pr_curves/all_models.png")
    plt.close()