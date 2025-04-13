# MLOps Project

This project is designed to implement a machine learning pipeline for processing data, training a deep learning model, and evaluating its performance. The project follows a modular approach, separating different functionalities into distinct modules for better organization and maintainability.

## Project Structure

```
mlops-project
├── data
│   ├── raw
│   │   └── filtered_file.csv
│   └── processed
├── src
│   ├── data_processing
│   │   ├── __init__.py
│   │   ├── outlier_treatment.py
│   │   ├── feature_scaling.py
│   │   └── data_split.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── model_training.py
│   │   ├── model_evaluation.py
│   │   └── model_saving.py
│   └── utils
│       ├── __init__.py
│       └── helper_functions.py
├── notebooks
│   └── exploratory_data_analysis.ipynb
├── tests
│   ├── test_data_processing.py
│   ├── test_model_training.py
│   └── test_model_evaluation.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd mlops-project
pip install -r requirements.txt
```

## Usage

1. **Data Processing**: Use the functions in the `src/data_processing` module to clean and preprocess the data. This includes outlier treatment, feature scaling, and splitting the dataset into training and testing sets.

2. **Model Training**: The `src/model/model_training.py` file contains the implementation of the deep learning model. You can modify the architecture and training parameters as needed.

3. **Model Evaluation**: After training the model, use the functions in `src/model/model_evaluation.py` to assess its performance using various metrics.

4. **Model Saving**: Save the trained model using the functionality provided in `src/model/model_saving.py`.

5. **Exploratory Data Analysis**: Utilize the Jupyter notebook located in `notebooks/exploratory_data_analysis.ipynb` for visualizing and understanding the dataset.

## Testing

Unit tests are provided in the `tests` directory to ensure the functionality of data processing, model training, and evaluation modules. You can run the tests using:

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.