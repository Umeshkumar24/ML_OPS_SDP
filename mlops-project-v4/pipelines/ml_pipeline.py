from zenml.pipelines import pipeline

@pipeline
def ml_pipeline(data_loader, preprocessor, feature_selector, model_trainer, evaluator):
    """
    Define the ML pipeline with ZenML.
    """
    # Step 1: Load data
    data = data_loader()

    # Step 2: Preprocess data
    X_train, X_test, y_train, y_test = preprocessor(data=data)

    # Step 3: Feature selection
    X_train_selected, X_test_selected = feature_selector(X_train=X_train, X_test=X_test, y_train=y_train)

    # Step 4: Train model
    model = model_trainer(X_train=X_train_selected, y_train=y_train)

    # Step 5: Evaluate model
    evaluator(model=model, X_test=X_test_selected, y_test=y_test)