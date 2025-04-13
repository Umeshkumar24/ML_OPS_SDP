import pandas as pd

def detect_outliers(data, column):
    mean = data[column].mean()
    std_dev = data[column].std()
    outlier_limit_upper = mean + 3 * std_dev
    outlier_limit_lower = mean - 3 * std_dev
    outliers = data[(data[column] > outlier_limit_upper) | (data[column] < outlier_limit_lower)]
    return outliers

def treat_outliers(data, column):
    mean = data[column].mean()
    std_dev = data[column].std()
    outlier_limit_upper = mean + 3 * std_dev
    outlier_limit_lower = mean - 3 * std_dev
    data = data[(data[column] <= outlier_limit_upper) & (data[column] >= outlier_limit_lower)]
    return data

def apply_outlier_treatment(data, columns):
    for column in columns:
        outliers = detect_outliers(data, column)
        print(f'Detected {len(outliers)} outliers in {column}.')
        data = treat_outliers(data, column)
    return data