"""
Breast Cancer Prediction - CSE3141 Predictive Analytics Mini Project
Run with:  python run_analysis.py
All charts are saved in outputs/plots and all tables in outputs/tables.
"""
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # save charts to files instead of opening windows

from src import data_loader, preprocessing, eda, features, models, evaluate


def main():
    # 1. Data collection
    df = data_loader.load_data()
    data_loader.basic_info(df)

    # 2. Preprocessing
    df = preprocessing.check_quality(df)
    feature_cols = [c for c in df.columns if c != "target"]
    preprocessing.iqr_outliers(df, feature_cols)

    # 3. EDA
    eda.run_eda(df)

    # 4. Feature selection and split
    selected = features.select_features(df)
    X_train, X_test, y_train, y_test = features.split_data(df, selected)

    # 5. Models, cross-validation, tuning
    all_models = models.build_models()
    models.cross_validate_models(all_models, X_train, y_train)
    all_models["Logistic Regression (tuned)"] = models.tune_logistic_regression(X_train, y_train)
    all_models["Random Forest (tuned)"] = models.tune_random_forest(X_train, y_train)

    # 6. Evaluation
    results = evaluate.evaluate_all(all_models, X_train, X_test, y_train, y_test)
    best_name, best = evaluate.best_model_report(results, all_models, X_test, y_test)
    evaluate.overfitting_check(best_name, best, X_train, X_test, y_train, y_test)
    evaluate.feature_importance_plot(all_models, selected)

    print("\nDone. Check the 'outputs' folder for plots and tables.")


if __name__ == "__main__":
    main()
