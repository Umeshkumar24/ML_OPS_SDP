import logging
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from src.pipelines.preprocessing.data_loader import load_and_clean_data
from src.pipelines.preprocessing.outlier_removal import OutlierRemover
from src.pipelines.preprocessing.feature_selection import FeatureSelector
from src.models.model_dnn import build_model
from src.models.train_eval import train_and_evaluate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # File paths
    file_path = 'mlops-project/data/filtered_file.csv'
    columns_to_clean = [
        'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN',
        'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE',
        'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE'
    ]

    # Step 1: Load and clean data
    logging.info("Loading and cleaning data...")
    data = load_and_clean_data(file_path)

    # Split data into features and target
    X = data.drop(['date', 'score'], axis=1)
    y = data['score']

    # Step 2: Split data into train and test sets
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing steps
    numeric_features = columns_to_clean
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('outlier_removal', OutlierRemover(X, columns=numeric_features)),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Define the full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('feature_selection', FeatureSelector(X, y, k=10)),
    ])
    model =  build_model(X_train, y_train)

    # Step 3: Train and evaluate the pipeline
    logging.info("Training and evaluating the pipeline...")
    train_and_evaluate(model, X_train, y_train, X_test, y_test)

if __name__ == "__main__":
    main()
