import pytest
import pandas as pd
from src.data_processing.outlier_treatment import remove_outliers
from src.data_processing.feature_scaling import scale_features
from src.data_processing.data_split import split_data

def test_remove_outliers():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 100],
        'B': [5, 6, 7, 8, 9]
    })
    cleaned_data = remove_outliers(data, 'A')
    assert len(cleaned_data) == 4  # One outlier should be removed
    assert 100 not in cleaned_data['A'].values

def test_scale_features():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9]
    })
    scaled_data = scale_features(data)
    assert scaled_data['A'].mean() == pytest.approx(0, abs=1e-2)
    assert scaled_data['B'].mean() == pytest.approx(0, abs=1e-2)

def test_split_data():
    data = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [5, 6, 7, 8, 9],
        'target': [0, 1, 0, 1, 0]
    })
    X_train, X_test, y_train, y_test = split_data(data, 'target', test_size=0.2)
    assert len(X_train) == 4
    assert len(X_test) == 1
    assert 'target' not in X_train.columns
    assert 'target' not in X_test.columns