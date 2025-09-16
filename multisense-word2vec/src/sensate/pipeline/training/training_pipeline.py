from sensate.pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from sensate.pipeline.preprocessing.corpus_pipeline import PairGenerator


class Trainer:
    def __init__(self, config=None):
        assert config is not None, "Config must be provided, the format is defined in sensate.schema"
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.training_sample_generator = PairGenerator()
    
    def fit(self, data):
        preprocessed_data = self.preprocessing_pipeline(data)
        corpus = self.training_sample_generator(corpus=preprocessed_data)
        print(f"Training model with data: {corpus}")

    def save_model(self, path=None):
        print(f"Model saved to {path if path else 'default_path'}")