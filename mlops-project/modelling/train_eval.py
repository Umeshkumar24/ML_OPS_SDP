from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2, verbose=1)
    test_loss = model.evaluate(X_test, y_test, batch_size=128)
    predictions = model.predict(X_test, batch_size=128)
    y_pred = np.argmax(predictions, axis=1)
    
    print(classification_report(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Precision:', precision_score(y_test, y_pred, average='weighted'))
    print('Recall:', recall_score(y_test, y_pred, average='weighted'))
    print('F1 Score:', f1_score(y_test, y_pred, average='weighted'))
    print(confusion_matrix(y_test, y_pred))
