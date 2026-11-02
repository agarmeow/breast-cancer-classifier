"""Step 6: Evaluation on the unseen test set, plus overfitting checks."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.model_selection import learning_curve
from src.config import PLOTS_DIR, TABLES_DIR
from src.models import get_cv


def evaluate_all(models, X_train, X_test, y_train, y_test):
    rows = []
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, prob)
        rows.append([name, accuracy_score(y_test, pred), precision_score(y_test, pred),
                     recall_score(y_test, pred), f1_score(y_test, pred), auc])
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label="%s (AUC=%.3f)" % (name, auc))
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "06_roc_curves.png"), dpi=120)
    plt.close(fig)

    results = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1",
                                          "AUC-ROC"]).round(4)
    print("\n" + "=" * 60)
    print("FINAL TEST-SET RESULTS")
    print("=" * 60)
    print(results.to_string(index=False))
    results.to_csv(os.path.join(TABLES_DIR, "final_results.csv"), index=False)
    return results


def best_model_report(results, models, X_test, y_test):
    best_name = results.sort_values(["F1", "AUC-ROC"], ascending=False).iloc[0]["Model"]
    best = models[best_name]
    pred = best.predict(X_test)
    print("\nBest model:", best_name)
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_test, pred), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix - " + best_name)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "07_confusion_matrix.png"), dpi=120)
    plt.close(fig)
    return best_name, best


def overfitting_check(best_name, best, X_train, X_test, y_train, y_test):
    """Compare train vs test F1 and draw a learning curve."""
    train_f1 = f1_score(y_train, best.predict(X_train))
    test_f1 = f1_score(y_test, best.predict(X_test))
    print("\n" + "=" * 60)
    print("OVERFITTING CHECK")
    print("=" * 60)
    print("Train F1: %.4f | Test F1: %.4f | Gap: %.4f" % (train_f1, test_f1, train_f1 - test_f1))
    print("A small gap means the model generalises well (no heavy overfitting).")

    sizes, tr, va = learning_curve(best, X_train, y_train, cv=get_cv(), scoring="f1",
                                   train_sizes=np.linspace(0.2, 1.0, 5), random_state=42,
                                   shuffle=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sizes, tr.mean(axis=1), "o-", label="Training F1")
    ax.plot(sizes, va.mean(axis=1), "o-", label="Validation F1")
    ax.set_xlabel("Training set size")
    ax.set_ylabel("F1 score")
    ax.set_title("Learning curve - " + best_name)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "08_learning_curve.png"), dpi=120)
    plt.close(fig)


def feature_importance_plot(models, selected):
    """Random Forest feature importances (if the model was trained)."""
    rf = models.get("Random Forest (tuned)")
    if rf is None:
        return
    imp = pd.Series(rf.named_steps["m"].feature_importances_, index=selected).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.plot.barh(ax=ax, color="#457b9d")
    ax.set_title("Feature importance - Random Forest")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "09_feature_importance.png"), dpi=120)
    plt.close(fig)
