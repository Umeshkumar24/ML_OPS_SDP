import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from preprocessing.outlier_removal import remove_outliers
from preprocessing.feature_selection import select_features_anova
from preprocessing.scaler import scale_data
from modelling.rf_feature_selector import select_features_rfe
from modelling.model_dnn import build_model
from modelling.train_eval import train_and_evaluate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and clean data
logging.info("Loading dataset...")
data = pd.read_csv('mlops-project/data/filtered_file.csv')
data.dropna(inplace=True)
data['score'] = data['score'].round().astype(int)
logging.info(f"Dataset loaded with shape: {data.shape}")

# Remove outliers
columns_to_clean = [
    'PRECTOT', 'PS', 'QV2M', 'T2M', 'T2MDEW', 'T2MWET', 'T2M_MAX', 'T2M_MIN',
    'T2M_RANGE', 'TS', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'WS10M_RANGE',
    'WS50M', 'WS50M_MAX', 'WS50M_MIN', 'WS50M_RANGE'
]
logging.info("Removing outliers...")
data = remove_outliers(data, columns_to_clean)
logging.info(f"Outlier removal completed. Data shape: {data.shape}")

# Prepare features
X = data.drop(['date', 'score'], axis=1)
y = data['score']
logging.info(f"Split into features and target. Features shape: {X.shape}, Target shape: {y.shape}")

# Identify categorical features
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
logging.info(f"Categorical features: {categorical_features}")
logging.info(f"Numerical features: {numerical_features}")

# Encoding categorical features
if categorical_features:
    logging.info("Applying OneHotEncoding to categorical features...")
    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough')
    X = preprocessor.fit_transform(X)
    logging.info(f"Encoding complete. Transformed feature shape: {X.shape}")
else:
    logging.info("No categorical features found. Skipping encoding.")

# Train-test split
logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Feature Selection
logging.info("Selecting features using RFE...")
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
selected_features = select_features_anova(X_train_df, y_train)
X_train_df = X_train_df[selected_features]
X_test_df = X_test_df[selected_features]
logging.info(f"Selected top {len(selected_features)} features.")

# Scale data
logging.info("Scaling features...")
X_train_scaled, X_test_scaled = scale_data(X_train_df, X_test_df)
logging.info("Scaling complete.")

# Build, train, and evaluate model
logging.info("Building and training the model...")
model = build_model(X_train_scaled.shape[1])
train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
logging.info("Model training and evaluation complete.")
