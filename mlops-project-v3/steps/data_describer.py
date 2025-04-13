import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def data_describer(data: pd.DataFrame, output_dir: str = "results"):
    """
    Analyze the dataset and generate visualizations.

    Args:
        data (pd.DataFrame): The dataset to analyze.
        output_dir (str): Directory to save the analysis results and visualizations.

    Returns:
        None
    """
    # Create output directory if it doesn't exist
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    data = data.drop('date', axis=1)
    # 1. Basic Dataset Information
    print("Dataset Information:")
    print(data.info())
    with open(os.path.join(output_dir, "dataset_info.txt"), "w") as f:
        data.info(buf=f)

    # 2. Summary Statistics
    print("\nSummary Statistics:")
    summary_stats = data.describe(include="all")
    print(summary_stats)
    summary_stats.to_csv(os.path.join(output_dir, "summary_statistics.csv"))

    # 3. Missing Values
    print("\nMissing Values:")
    missing_values = data.isnull().sum()
    print(missing_values)
    missing_values.to_csv(os.path.join(output_dir, "missing_values.csv"))

    # 4. Correlation Matrix
    print("\nCorrelation Matrix:")
    correlation_matrix = data.corr()
    print(correlation_matrix)
    correlation_matrix.to_csv(os.path.join(output_dir, "correlation_matrix.csv"))

    # Plot the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
    plt.close()

    # 5. Distribution of Each Feature
    print("\nGenerating feature distributions...")
    for column in data.select_dtypes(include=["float64", "int64"]).columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.savefig(os.path.join(output_dir, f"{column}_distribution.png"))
        plt.close()

    # 6. Boxplots for Outlier Detection
    print("\nGenerating boxplots for numerical features...")
    for column in data.select_dtypes(include=["float64", "int64"]).columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=data[column])
        plt.title(f"Boxplot of {column}")
        plt.xlabel(column)
        plt.savefig(os.path.join(output_dir, f"{column}_boxplot.png"))
        plt.close()

    print(f"\nData analysis and visualizations saved in '{output_dir}' directory.")