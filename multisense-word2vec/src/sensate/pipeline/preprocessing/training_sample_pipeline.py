import pandas as pd
from typing import List, Dict
from sensate.pipeline.preprocessing.corpus_pipeline import PairGenerator, BERTEmbeddingGenerator
from sensate.schema.config_schema import BaseTableEntry


class TrainingSampleGenerator:
    def __init__(self, window_size: int = 2, foundation_model_name: str = None):
        self.pair_generator = PairGenerator(window_size=window_size)
        self.embedding_generator = BERTEmbeddingGenerator(foundation_model_name=foundation_model_name)
        self.next_id = 0
        self.word_id_counter = 0
        self.query_id_counter = 0
        self.embedding_id_counter = 0

    def __call__(self, corpus: List[List[str]]) -> tuple:
        """
        4 tables:
        - vocab_table: DataFrame with columns [word, id]
        - query_table: DataFrame with columns [id, sql_query]
        - embedding_table: DataFrame with columns [id, embedding]
        - base_table: DataFrame with columns [id, center_word_id, context_word_id, embedding_id, sql_query_id]
        """
        # Step 1: Generate pairs (center-context) and contextual BERT embeddings
            
        pairs = self.pair_generator(corpus)
        embeddings = self.embedding_generator(corpus)  # [{token: np.array(768)}, ...]

        vocab_dict, query_dict, embedding_dict, base_list = {}, {}, {}, []
        embedding_id_map = {}  # (token, query_id) -> embedding_id

        # Step 2: Build vocabulary
        unique_words = set(word for sentence in corpus for word in sentence)
        for word in unique_words:
            vocab_dict[word] = self.word_id_counter
            self.word_id_counter += 1

        # Step 3: Build query table
        for i, sentence in enumerate(corpus):
            query_dict[i] = " ".join(sentence)
            self.query_id_counter += 1

        # Step 4: Build embedding table (unique per token per query)
        for query_id, emb_dict in enumerate(embeddings):
            for token, emb in emb_dict.items():
                embedding_dict[self.embedding_id_counter] = emb
                embedding_id_map[(token, query_id)] = self.embedding_id_counter
                self.embedding_id_counter += 1

        # Step 5: Build base table using IDs only
        for query_id, sentence_pairs in enumerate(pairs):
            for center, context in sentence_pairs:
                if (center, query_id) not in embedding_id_map:
                    continue

                base_entry = {
                    'id': self.next_id,
                    'center_word_id': vocab_dict[center],
                    'context_word_id': vocab_dict[context],
                    'embedding_id': embedding_id_map[(center, query_id)],
                    'sql_query_id': query_id
                }
                base_list.append(base_entry)
                self.next_id += 1

        # Convert to pandas DataFrames
        vocab_table = pd.DataFrame(list(vocab_dict.items()), columns=['word', 'id'])
        query_table = pd.DataFrame(list(query_dict.items()), columns=['id', 'sql_query'])
        embedding_table = pd.DataFrame(list(embedding_dict.items()), columns=['id', 'embedding'])
        base_table = pd.DataFrame(base_list)

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
