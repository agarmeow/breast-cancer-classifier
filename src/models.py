"""Step 5: Model definitions, cross-validation and hyperparameter tuning."""
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from src.config import RANDOM_STATE, CV_FOLDS, TABLES_DIR


def get_cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_models():
    """Pipelines include the scaler so scaling is learned from training folds only (no leakage)."""
    return {
        "Logistic Regression": Pipeline([("sc", StandardScaler()),
                                         ("m", LogisticRegression(max_iter=1000))]),
        "LDA": Pipeline([("sc", StandardScaler()), ("m", LinearDiscriminantAnalysis())]),
        "QDA": Pipeline([("sc", StandardScaler()),
                         ("m", QuadraticDiscriminantAnalysis(reg_param=0.1))]),
        "Random Forest": Pipeline([("sc", StandardScaler()),
                                   ("m", RandomForestClassifier(n_estimators=100,
                                                                random_state=RANDOM_STATE))]),
    }


def cross_validate_models(models, X_train, y_train):
    cv = get_cv()
    rows = []
    for name, model in models.items():
        f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
        auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        rows.append([name, f1.mean(), f1.std(), auc.mean()])
    table = pd.DataFrame(rows, columns=["Model", "CV F1 mean", "CV F1 std", "CV AUC mean"]).round(4)
    print("\n" + "=" * 60)
    print("%d-FOLD CROSS-VALIDATION (training data only)" % CV_FOLDS)
    print("=" * 60)
    print(table.to_string(index=False))
    table.to_csv(os.path.join(TABLES_DIR, "cross_validation.csv"), index=False)
    return table


def tune_logistic_regression(X_train, y_train):
    """Grid search over regularisation strength C."""
    pipe = Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000))])
    grid = GridSearchCV(pipe, {"m__C": [0.01, 0.1, 1, 10, 100]}, cv=get_cv(), scoring="f1")
    grid.fit(X_train, y_train)
    print("\n" + "=" * 60)
    print("GRID SEARCH - Logistic Regression")
    print("=" * 60)
    print("Best parameters:", grid.best_params_, "| Best CV F1:", round(grid.best_score_, 4))
    return grid.best_estimator_


def tune_random_forest(X_train, y_train):
    """Random search over a few Random Forest settings."""
    pipe = Pipeline([("sc", StandardScaler()),
                     ("m", RandomForestClassifier(random_state=RANDOM_STATE))])
    space = {"m__n_estimators": [50, 100, 200],
             "m__max_depth": [3, 5, 8, None],
             "m__min_samples_leaf": [1, 2, 4]}
    rs = RandomizedSearchCV(pipe, space, n_iter=8, cv=get_cv(), scoring="f1",
                            random_state=RANDOM_STATE)
    rs.fit(X_train, y_train)
    print("\n" + "=" * 60)
    print("RANDOM SEARCH - Random Forest")
    print("=" * 60)
    print("Best parameters:", rs.best_params_, "| Best CV F1:", round(rs.best_score_, 4))
    return rs.best_estimator_
