from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def select_features_rfe(X, y, n_features=15):
    model = RandomForestClassifier(n_estimators=10)
    rfe = RFE(model, n_features_to_select=n_features)
    rfe.fit(X, y)
    return X.columns[rfe.get_support()]
