import logging

def OutlierRemover(data, columns):
    logging.info("Removing outliers...")
    for column in columns:
        q1 = data[column].quantile(0.25)
        q3 = data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    logging.info(f"Outlier removal completed. Data shape: {data.shape}")
    return data
