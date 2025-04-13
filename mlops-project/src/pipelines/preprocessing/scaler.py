import logging
from sklearn.preprocessing import StandardScaler

def scale_data(X_train, X_test):
    logging.info("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    logging.info("Scaling complete.")
    return X_train_scaled, X_test_scaled
