import pandas as pd

def remove_outliers(data, columns):
    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        data = data[(data[col] >= mean - 3*std) & (data[col] <= mean + 3*std)]
    return data
