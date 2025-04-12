def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1, validation_split=0.1)
    loss, mae = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {loss}, MAE: {mae}")
