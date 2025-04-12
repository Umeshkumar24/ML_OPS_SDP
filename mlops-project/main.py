import pandas as pd
from sklearn.model_selection import train_test_split

from preprocessing.outlier_removal import remove_outliers
from preprocessing.feature_selection import select_features_anova
from preprocessing.scaler import scale_data
from modelling.rf_feature_selector import select_features_rfe
from modelling.model_dnn import build_model
from modelling.train_eval import train_and_evaluate

# Load and preprocess data
data = pd.read_csv('./data/filtered_file.csv')
data.dropna(inplace=True)
data['score'] = data['score'].round().astype(int)

columns_to_clean = [
    'PRECTOT','PS','QV2M','T2M','T2MDEW','T2MWET','T2M_MAX','T2M_MIN',
    'T2M_RANGE','TS','WS10M','WS10M_MAX','WS10M_MIN','WS10M_RANGE',
    'WS50M','WS50M_MAX','WS50M_MIN','WS50M_RANGE'
]
data = remove_outliers(data, columns_to_clean)

X = data.drop(['date'], axis=1)
y = data['score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Selection
selected_features = select_features_rfe(X_train, y_train)
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# Scale
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Build and Train
model = build_model(X_train_scaled.shape[1])
train_and_evaluate(model, X_train_scaled, y_train, X_test_scaled, y_test)
