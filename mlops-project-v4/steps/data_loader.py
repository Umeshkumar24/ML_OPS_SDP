from zenml.steps import step
import pandas as pd

@step
def data_loader() -> pd.DataFrame:
    """
    Load and clean the dataset.
    """
    file_path = 'data/filtered_file.csv'
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)
    data['score'] = data['score'].round().astype(int)
    return data