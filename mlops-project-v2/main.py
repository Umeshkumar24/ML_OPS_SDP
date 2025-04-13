import os
import pandas as pd
from src.pipelines.data_processing.outlier_treatment import treat_outliers
from src.pipelines.data_processing.feature_scaling import scale_features
from src.pipelines.data_processing.data_split import split_data
from src.model.model_training import train_model
from src.model.model_evaluation import evaluate_model
from src.model.model_saving import save_model
from src.pipelines.data_analysis.data_description import describe_and_analyze_data


def main():
    # Paths
    raw_data_path = "mlops-project/mlops-project/data/raw/filtered_file.csv"
    processed_data_path = "mlops-project/mlops-project/data/raw/filtered_file.csv"
    model_save_path = "models/draught_model.h5"

    # Ensure the processed data directory exists
    os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)

    # Step 1: Data Processing
    print("Step 1: Processing Data...")
    raw_data = pd.read_csv(raw_data_path)

    # Step 1.1: Data Description and Analysis
    print("Step 1.1: Describing and Analyzing Data...")
    describe_and_analyze_data(raw_data)

    for column in raw_data.columns:
        raw_data[column] = treat_outliers(raw_data, column)
    raw_data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")

    # Step 2: Splitting Data
    print("Step 2: Splitting Data...")
    X_train, X_test, y_train, y_test = split_data(raw_data , 'score', test_size=0.2, random_state=42)

    # Step 3: Feature Scaling
    print("Step 3: Scaling Features...")
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Step 4: Model Training
    print("Step 4: Training Model...")
    model, history = train_model(X_train_scaled, y_train)

    # Step 5: Model Evaluation
    print("Step 5: Evaluating Model...")
    evaluate_model(model, X_test_scaled, y_test)

    # Step 6: Save Model
    print("Step 6: Saving Model...")
    save_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()