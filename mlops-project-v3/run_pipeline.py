import mlflow
import mlflow.sklearn
from steps.data_loader import data_loader
from steps.preprocessor import preprocessor
from steps.scaler import scaler 
from steps.feature_selector import feature_selector
from steps.model_trainer import model_trainer
from steps.evaluator import evaluator
from steps.class_balancer import balance_classes
from steps.data_describer import data_describer  

# Start an MLflow experiment
mlflow.set_experiment("mlops_project_experiment")

with mlflow.start_run():
    # Step 1: Load data
    data = data_loader()
    mlflow.log_param("data_shape", data.shape)

    # Step 2: Analyze the dataset
    data_describer(data=data, output_dir="results")

    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test = preprocessor(data=data)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # Step 4: Balance classes in the training data
    X_train_balanced, y_train_balanced = balance_classes(
        X_train=X_train,
        y_train=y_train,
        method="undersample",  # Use "oversample" or "undersample"
        algorithm="nearmiss"     # Use "smote" or "nearmiss"
    )

    # Step 5: Scale data
    X_train_scaled, X_test_scaled = scaler(X_train=X_train_balanced, X_test=X_test)

    # Step 6: Feature selection
    X_train_selected, X_test_selected = feature_selector(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train_balanced,
        method="anova",  # Use "anova", "rfe", "dti", or "correlation"
        k=10             # Number of top features to select
    )

    # Step 7: Train model
    model = model_trainer(X_train=X_train_selected, y_train=y_train_balanced)

    # Log the trained model with an input example
    input_example = X_train_selected[:1]  # Example input
    mlflow.sklearn.log_model(model, "model", input_example=input_example)

    # Step 8: Evaluate model
    evaluator(model=model, X_test=X_test_selected, y_test=y_test, output_dir="results")