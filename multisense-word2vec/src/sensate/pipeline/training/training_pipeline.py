from sensate.pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline


class Trainer:
    def __init__(self, config=None):
        assert config is not None, "Config must be provided, the format is defined in sensate.schema"
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline()
    
    def fit(self, data):
        preprocessed_data = self.preprocessing_pipeline(data)
        print(f"Training model with data: {preprocessed_data}")

    def save_model(self, path=None):
        print(f"Model saved to {path if path else 'default_path'}")