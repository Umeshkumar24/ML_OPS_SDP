import logging
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd

def FeatureSelector(X, y , k):
    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    logging.info("Selecting features using ANOVA...")
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = selector.get_support(indices=True)
    logging.info(f"Selected top {len(selected_features)} features.")
    return pd.DataFrame(X_train_selected), pd.DataFrame(X_test_selected), y_train, y_test, selected_features
