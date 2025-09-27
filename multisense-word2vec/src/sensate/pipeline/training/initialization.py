class SenseEmbeddingsInitializer:
    def __init__(self, base_table, vocab, config):
        self.base_table = base_table
        self.vocab = vocab
        self.config = config

    def __call__(self):
        # Initialize sense embeddings based on base_table and vocab
        print(f"Initializing sense embeddings for {len(self.vocab)} words.")
        # Placeholder for actual initialization logic
        return {"sense_embeddings": "initialized"}
    
class OutputEmbeddingsInitializer:
    # TODO: AQO-11
    def __init__(self, vocab, config):
        self.vocab = vocab
        self.config = config

    def __call__(self):
        # Initialize output embeddings based on vocab
        print(f"Initializing output embeddings for {len(self.vocab)} words.")
        # Placeholder for actual initialization logic
        return {"output_embeddings": "initialized"}
    
class ProjectionMatricesInitializer:
    # TODO: AQO-11
    def __init__(self):
        pass

    def __call__(self):
        # Initialize projection matrices based on vocab
        print(f"Initializing projection matrices")
        # Placeholder for actual initialization logic
        return {"projection_matrices": "initialized"}