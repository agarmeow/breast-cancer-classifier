"""Step 2: Data preprocessing - missing values, duplicates, outliers (IQR)."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PLOTS_DIR, TABLES_DIR
import os


def check_quality(df):
    """Report missing values and duplicate rows."""
    missing = int(df.isnull().sum().sum())
    dups = int(df.duplicated().sum())
    print("\n" + "=" * 60)
    print("DATA QUALITY CHECK")
    print("=" * 60)
    print("Missing values:", missing)
    print("Duplicate rows:", dups)
    if missing > 0:
        df = df.fillna(df.median(numeric_only=True))
        print("Missing values filled with column medians.")
    if dups > 0:
        df = df.drop_duplicates()
        print("Duplicate rows removed.")
    return df


def iqr_outliers(df, features):
    """Count outliers per feature using the IQR rule (1.5 x IQR)."""
    q1, q3 = df[features].quantile(0.25), df[features].quantile(0.75)
    iqr = q3 - q1
    mask = (df[features] < q1 - 1.5 * iqr) | (df[features] > q3 + 1.5 * iqr)
    counts = mask.sum().sort_values(ascending=False)
    print("\n" + "=" * 60)
    print("OUTLIERS (IQR RULE)")
    print("=" * 60)
    print(counts.head(10))
    counts.to_csv(os.path.join(TABLES_DIR, "outlier_counts.csv"), header=["outliers"])

    top = counts.index[:4].tolist()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, top):
        sns.boxplot(data=df, y=col, ax=ax, color="#8ecae6")
        ax.set_title(col)
    fig.suptitle("Box plots of the features with the most outliers")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "02_outlier_boxplots.png"), dpi=120)
    plt.close(fig)

    print("\nDecision: outliers are genuine large tumours, not errors, so they are KEPT.")
    return counts
