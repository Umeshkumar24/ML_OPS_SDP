from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def select_features_rfe(X, y, num_features=10):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(model, n_features_to_select=num_features)
    X_selected = rfe.fit_transform(X, y)
    selected_columns = X.columns[rfe.get_support()]
    return X_selected, selected_columns
