import pandas as pd
import logging

def load_and_clean_data(file_path):
    logging.info("Loading dataset...")
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    data['score'] = data['score'].round().astype(int)
    logging.info(f"Dataset loaded with shape: {data.shape}")
    return data