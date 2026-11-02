"""Step 4: Feature selection and the train-test split."""
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from src.config import (N_SELECTED_FEATURES, RANDOM_STATE, TEST_SIZE, TABLES_DIR)


def select_features(df, k=N_SELECTED_FEATURES):
    """Keep the k features with the highest ANOVA F-score against the target."""
    features = [c for c in df.columns if c != "target"]
    selector = SelectKBest(f_classif, k=k).fit(df[features], df["target"])
    scores = pd.DataFrame({"feature": features, "f_score": selector.scores_}) \
        .sort_values("f_score", ascending=False)
    scores.to_csv(os.path.join(TABLES_DIR, "feature_scores.csv"), index=False)
    selected = scores["feature"].head(k).tolist()
    print("\n" + "=" * 60)
    print("FEATURE SELECTION (ANOVA F-test, top %d)" % k)
    print("=" * 60)
    print(scores.head(k).to_string(index=False))
    return selected


def split_data(df, selected):
    """Stratified 80/20 split so both sets keep the same class balance."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[selected], df["target"], test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=df["target"])
    print("\nTrain size:", X_train.shape, "| Test size:", X_test.shape)
    return X_train, X_test, y_train, y_test
