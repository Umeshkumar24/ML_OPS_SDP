import pytest
from src.pipelines.pipeline import DataPipeline

def test_data_pipeline_initialization():
    pipeline = DataPipeline()
    assert pipeline is not None

def test_data_ingestion():
    pipeline = DataPipeline()
    data = pipeline.ingest_data()
    assert data is not None
    assert len(data) > 0

def test_data_processing():
    pipeline = DataPipeline()
    raw_data = pipeline.ingest_data()
    processed_data = pipeline.process_data(raw_data)
    assert processed_data is not None
    assert len(processed_data) > 0

def test_model_training():
    pipeline = DataPipeline()
    raw_data = pipeline.ingest_data()
    processed_data = pipeline.process_data(raw_data)
    model = pipeline.train_model(processed_data)
    assert model is not None

def test_model_evaluation():
    pipeline = DataPipeline()
    raw_data = pipeline.ingest_data()
    processed_data = pipeline.process_data(raw_data)
    model = pipeline.train_model(processed_data)
    evaluation_results = pipeline.evaluate_model(model, processed_data)
    assert evaluation_results is not None
    assert 'accuracy' in evaluation_results
    assert evaluation_results['accuracy'] >= 0.5  # Assuming a baseline accuracy of 50%