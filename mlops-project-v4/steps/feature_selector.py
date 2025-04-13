from zenml.steps import step
from sklearn.feature_selection import SelectKBest, f_classif

@step
def feature_selector(X_train, X_test, y_train):
    """
    Perform feature selection using ANOVA.
    """
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected