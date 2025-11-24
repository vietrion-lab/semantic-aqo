import pandas as pd
import numpy as np
from array import array
from typing import List
from sensate.pipeline.preprocessing.corpus_pipeline import PairGenerator, BERTEmbeddingGenerator

class TrainingSampleGenerator:
    def __init__(self, window_size: int = 2, foundation_model_name: str = None, ipca=None):
        self.pair_generator = PairGenerator(window_size=window_size)
        self.embedding_generator = BERTEmbeddingGenerator(foundation_model_name=foundation_model_name, ipca=ipca)
        self.next_id = 0
        self.word_id_counter = 0
        self.query_id_counter = 0
        self.embedding_id_counter = 0

    def __call__(self, corpus: List[List[str]]) -> tuple:
        """
        4 tables:
        - vocab_table: DataFrame with columns [word, id]
        - query_table: DataFrame with columns [id, sql_query, sql_length]
        - embedding_table: DataFrame with columns [id, embedding]
        - base_table: DataFrame with columns [id, center_word_id, center_pos, context_word_id, embedding_id, sql_query_id]
        """
        vocab_dict = {}
        vocab_records = []  # (word, id)
        query_records = []  # (id, sql_query, sql_length)
        embedding_rows = []  # (embedding_id, embedding_vector)

        # Base table columns stored as compact arrays to minimize RAM
        center_word_ids = array('I')
        center_positions = array('I')
        context_word_ids = array('I')
        embedding_ids = array('I')
        query_ids = array('I')

        embedding_iter = self.embedding_generator.iter_embeddings(corpus)

        for query_id, sentence_embeddings in enumerate(embedding_iter):
            sentence = corpus[query_id]
            query_records.append((query_id, " ".join(sentence), len(sentence)))

            token_vocab_ids = []
            token_embedding_ids = []

            if len(sentence) != len(sentence_embeddings):
                raise ValueError(f"Embedding/token length mismatch at query {query_id}")

            for token_idx, (token, embedding) in enumerate(zip(sentence, sentence_embeddings)):
                if token not in vocab_dict:
                    token_id = self.word_id_counter
                    vocab_dict[token] = token_id
                    vocab_records.append((token, token_id))
                    self.word_id_counter += 1
                else:
                    token_id = vocab_dict[token]

                token_vocab_ids.append(token_id)
                embedding_rows.append((self.embedding_id_counter, embedding))
                token_embedding_ids.append(self.embedding_id_counter)
                self.embedding_id_counter += 1

            sentence_pairs = self.pair_generator.generate_center_context_pair(sentence)
            for center_idx, context_idx in sentence_pairs:
                center_word_ids.append(token_vocab_ids[center_idx])
                center_positions.append(center_idx)
                context_word_ids.append(token_vocab_ids[context_idx])
                embedding_ids.append(token_embedding_ids[center_idx])
                query_ids.append(query_id)

        vocab_table = pd.DataFrame(vocab_records, columns=['word', 'id']).sort_values('id').reset_index(drop=True)
        query_table = pd.DataFrame(query_records, columns=['id', 'sql_query', 'sql_length']).sort_values('id').reset_index(drop=True)
        embedding_table = pd.DataFrame(embedding_rows, columns=['id', 'embedding'])

        # Convert base arrays to numpy int32 for pandas compatibility
        center_word_ids_np = np.frombuffer(center_word_ids, dtype=np.uint32).astype(np.int32)
        center_positions_np = np.frombuffer(center_positions, dtype=np.uint32).astype(np.int32)
        context_word_ids_np = np.frombuffer(context_word_ids, dtype=np.uint32).astype(np.int32)
        embedding_ids_np = np.frombuffer(embedding_ids, dtype=np.uint32).astype(np.int32)
        query_ids_np = np.frombuffer(query_ids, dtype=np.uint32).astype(np.int32)

        base_table = pd.DataFrame({
            'id': np.arange(len(center_word_ids_np), dtype=np.int32),
            'center_word_id': center_word_ids_np,
            'center_pos': center_positions_np,
            'context_word_id': context_word_ids_np,
            'embedding_id': embedding_ids_np,
            'sql_query_id': query_ids_np
        })

        return vocab_table, query_table, embedding_table, base_table

# if __name__ == "__main__":
#     sample_corpus = [
#         ["SELECT", "<TAB>", "WHERE", "<COL>", "=", "<STR>"],
#         ["SELECT", "<COL>", "FROM", "<TAB>", "JOIN", "<TAB>", "ON", "<TAB>", ".", "<COL>", "=", "<TAB>", ".", "<COL>"]
#     ]

#     training_sample_generator = TrainingSampleGenerator(window_size=2)
#     vocab_table, query_table, embedding_table, base_table = training_sample_generator(corpus=sample_corpus)

#     # Print tables
#     print_pretty_tables(vocab_table, query_table, embedding_table, base_table)
