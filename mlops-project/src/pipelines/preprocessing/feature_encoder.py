import logging
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def encode_features(data):
    X = data.drop(['date', 'score'], axis=1)
    y = data['score']
    logging.info(f"Split into features and target. Features shape: {X.shape}, Target shape: {y.shape}")

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_features:
        logging.info("Applying OneHotEncoding to categorical features...")
        preprocessor = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')
        X = preprocessor.fit_transform(X)
        logging.info(f"Encoding complete. Transformed feature shape: {X.shape}")
    else:
        logging.info("No categorical features found. Skipping encoding.")
    return X, y