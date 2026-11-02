"""Step 1: Data collection. The dataset ships with scikit-learn (no download)."""
import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    """Return a DataFrame with 30 features and a 'target' column (1 = malignant, 0 = benign)."""
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # sklearn encodes 0 = malignant, 1 = benign. Flip it so 1 = malignant (the event we predict).
    df["target"] = 1 - df["target"]
    return df


def basic_info(df):
    """Print a quick overview of the dataset."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print("Shape (rows, columns):", df.shape)
    counts = df["target"].value_counts().rename({0: "Benign", 1: "Malignant"})
    print("\nClass distribution:\n", counts)
    print("\nFirst 5 rows:\n", df.head())
    return counts
