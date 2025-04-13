'""'

from pipelines.ml_pipeline import ml_pipeline
from steps.data_loader import data_loader
from steps.preprocessor import preprocessor
from steps.feature_selector import feature_selector
from steps.model_trainer import model_trainer
from steps.evaluator import evaluator

# Instantiate and run the pipeline
pipeline = ml_pipeline(
    data_loader=data_loader(),
    preprocessor=preprocessor(),
    feature_selector=feature_selector(),
    model_trainer=model_trainer(),
    evaluator=evaluator()
)
pipeline.run()