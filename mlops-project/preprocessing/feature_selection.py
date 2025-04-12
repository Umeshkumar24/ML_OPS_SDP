from sklearn.feature_selection import SelectKBest, f_classif

def select_features_anova(X_train, y_train, k=13):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X_train, y_train)
    return X_train.columns[selector.get_support()]
