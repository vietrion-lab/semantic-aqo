from sensate.pipeline.preprocessing.corpus_pipeline import PairGenerator
from sensate.pipeline.preprocessing.corpus_pipeline import BERTEmbeddingGenerator


class TrainingSampleGenerator:
    def __init__(self, window_size: int = 2):
        self.pair_generator = PairGenerator(window_size=window_size)
        self.embedding_generator = BERTEmbeddingGenerator()
    
    def __call__(self, corpus: list) -> list:
        pairs = self.pair_generator(corpus)
        embeddings = self.embedding_generator(corpus)
        # TODO: AQO-9 - Merge job
        pass
    
# Test code with multiple sample data
# if __name__ == "__main__":
#     # Multiple sample inputs based on provided figures
#     sample_corpus = [
#         ["SELECT", "<TAB>", "WHERE", "<COL>", "=", '<STR>'],
#         ["SELECT", "<COL>", "FROM", "<TAB>", "JOIN", "<TAB>", "ON", "<TAB>", ".", "<COL>", "=", "<TAB>", ".", "<COL>"]
#     ]
#     # Initialize TrainingSampleGenerator with default window_size=2
#     training_sample_generator = TrainingSampleGenerator(window_size=2)
#     base_table, vocab_table, embedding_table, query_table = training_sample_generator(corpus=sample_corpus)
    
#     print(f"Base Table: {base_table}")
#     print(f"Vocab Table: {vocab_table}")
#     print(f"Embedding Table: {embedding_table}")
#     print(f"Query Table: {query_table}")