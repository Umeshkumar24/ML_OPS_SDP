from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

def select_features_rfe(X, y, n_features_to_select=10):
    model = RandomForestClassifier(n_estimators=10)
    selector = RFE(model, n_features_to_select=n_features_to_select)
    selector = selector.fit(X, y)
    return X.columns[selector.support_]

def select_features_anova(X, y, k=10):
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]