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
        self.training_sample_generator = TrainingSampleGenerator(
            window_size=config.training.window_size, 
            foundation_model_name=config.foundation.foundation_model
        )

    def fit(self, data):
        # Step 1: Prepare training samples
        preprocessed_data = self.preprocessing_pipeline(data)
        vocab_table, query_table, embedding_table, base_table = self.training_sample_generator(corpus=preprocessed_data)
        print(f"Corpus length: {len(base_table)}, Vocab size: {len(vocab_table)}")
        
        print("=" * 20)
        print(f"Base Table: \n{base_table.head()}")
        print(f"Vocab Table: \n{vocab_table.head()}")
        print(f"Embedding Table: \n{embedding_table.head()}")
        print(f"Query Table: \n{query_table.head()}")
        print("=" * 20)

        # Step 2: Init matrices
        sense_embeddings = SenseEmbeddingsInitializer(
            base_table=base_table, 
            vocab=vocab_table, 
            embedding=embedding_table,
            embedding_dim=self.config.training.embedding_dim,
            num_senses=self.config.training.num_senses
        )()
        print(f"Sense Embeddings: \n{sense_embeddings.head()}")
        if len(sense_embeddings) > 0:
            print(f"Sense Embedding Shape: {len(sense_embeddings['embedding'].iloc[0])} dims")
        print("=" * 20)

        output_embeddings = OutputEmbeddingsInitializer(
            vocab=vocab_table, 
            embedding_dim=self.config.training.embedding_dim
        )()
        print(f"Output Embeddings: {output_embeddings.shape}")
        print("=" * 20)
        
        projection_matrix = ProjectionMatricesInitializer(
            bert_dim=self.config.training.distill_dim,
            embedding_dim=self.config.training.embedding_dim
        )()
        print(f"Projection Matrix: {projection_matrix.shape}")
        print("=" * 20)
        
        # Step 3: Training loop

    def save_model(self, path=None):
        print(f"Model saved to {path if path else 'default_path'}")