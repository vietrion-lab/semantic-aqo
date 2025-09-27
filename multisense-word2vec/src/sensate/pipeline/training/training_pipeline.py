from sensate.pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from sensate.pipeline.preprocessing.training_sample_pipeline import TrainingSampleGenerator
from sensate.pipeline.training.initialization import (
    SenseEmbeddingsInitializer, 
    OutputEmbeddingsInitializer, 
    ProjectionMatricesInitializer
)


class Trainer:
    def __init__(self, config=None):
        assert config is not None, "Config must be provided, the format is defined in sensate.schema"
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.training_sample_generator = TrainingSampleGenerator(window_size=config.window_size)
     
    def fit(self, data):
        # Step 1: Prepare training samples
        preprocessed_data = self.preprocessing_pipeline(data)
        base_table, vocab_table, embedding_table, query_table = self.training_sample_generator(corpus=preprocessed_data)
        print(f"Corpus length: {len(base_table)}, Vocab size: {len(vocab_table)}")

        # Step 2: Init matrices
        sense_embeddings = SenseEmbeddingsInitializer(base_table=base_table, vocab=vocab_table, config=self.config)()
        output_embeddings = OutputEmbeddingsInitializer(vocab=vocab_table, config=self.config)()
        projection_matrices = ProjectionMatricesInitializer(vocab=vocab_table, config=self.config)()
        

        # Step 3: Training loop

    def save_model(self, path=None):
        print(f"Model saved to {path if path else 'default_path'}")