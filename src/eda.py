"""Step 3: Exploratory Data Analysis - statistics and visualisations."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import PLOTS_DIR, TABLES_DIR

sns.set(style="whitegrid")


def summary_statistics(df):
    """Mean, median, variance, standard deviation for key features."""
    cols = ["mean radius", "mean texture", "mean area", "mean concavity"]
    stats = df[cols].agg(["mean", "median", "var", "std"]).T.round(3)
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(stats)
    stats.to_csv(os.path.join(TABLES_DIR, "summary_statistics.csv"))
    return stats


def class_distribution_plot(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].value_counts().rename({0: "Benign", 1: "Malignant"})
    ax.bar(counts.index, counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Class distribution")
    ax.set_ylabel("Number of samples")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "01_class_distribution.png"), dpi=120)
    plt.close(fig)

    pie_fig, pie_ax = plt.subplots(figsize=(4.5, 4.5))
    pie_ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%", colors=["#2a9d8f", "#e76f51"])
    pie_ax.set_title("Class share")
    pie_fig.tight_layout()
    pie_fig.savefig(os.path.join(PLOTS_DIR, "01b_class_pie.png"), dpi=120)
    plt.close(pie_fig)


def histograms(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["mean radius", "mean texture", "mean concavity"]):
        sns.histplot(data=df, x=col, hue="target", kde=True, ax=ax)
        ax.set_title("Histogram: " + col)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "03_histograms.png"), dpi=120)
    plt.close(fig)


def scatter_plot(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.scatterplot(data=df, x="mean radius", y="mean concavity", hue="target", ax=ax)
    ax.set_title("Scatter: mean radius vs mean concavity")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "04_scatter.png"), dpi=120)
    plt.close(fig)


def correlation_heatmap(df):
    cols = ["mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
            "mean compactness", "mean concavity", "mean concave points", "target"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "05_correlation_heatmap.png"), dpi=120)
    plt.close(fig)


def run_eda(df):
    summary_statistics(df)
    class_distribution_plot(df)
    histograms(df)
    scatter_plot(df)
    correlation_heatmap(df)
    print("EDA plots saved to outputs/plots/")
