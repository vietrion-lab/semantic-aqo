import pandas as pd
from typing import List, Dict
from sensate.pipeline.preprocessing.corpus_pipeline import PairGenerator, BERTEmbeddingGenerator
from sensate.schema.config_schema import BaseTableEntry
from tqdm import tqdm
import numpy as np

import torch

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
        # Step 1: Generate pairs (center-context) and contextual BERT embeddings
            
        pairs = self.pair_generator(corpus)
        embeddings = self.embedding_generator(corpus)  # [{token: np.array(768)}, ...]

        # Free up BERT model memory immediately
        del self.embedding_generator
        torch.cuda.empty_cache()

        vocab_dict, query_dict = {}, {}
        embedding_list = []  # Use list instead of dict for memory efficiency
        embedding_id_map = {}  # (token, query_id) -> embedding_id
        token_position_map = {}  # (token, query_id) -> position in query
        base_list = []

        # Step 2: Build vocabulary
        print("📚 Building vocabulary...")
        unique_words = set(word for sentence in corpus for word in sentence)
        for word in tqdm(unique_words, desc="Step 2: Building vocab", unit="word"):
            vocab_dict[word] = self.word_id_counter
            self.word_id_counter += 1

        # Step 3: Build query table
        print("📝 Building query table...")
        query_list = []
        for i in tqdm(range(len(corpus)), desc="Step 3: Building queries", unit="query"):
            sentence = corpus[i]
            query_list.append({
                'id': i,
                'sql_query': " ".join(sentence),
                'sql_length': len(sentence)
            })
            self.query_id_counter += 1
            # Track position of each token in the query
            for pos, token in enumerate(sentence):
                token_position_map[(token, i)] = pos

        # Step 4: Build embedding table (unique per token per query) - memory efficient
        print("🔢 Building embedding table...")
        for query_id in tqdm(range(len(embeddings)), desc="Step 4: Building embeddings", unit="query"):
            emb_dict = embeddings[query_id]
            for token, emb in emb_dict.items():
                # Append to list instead of dict (more memory efficient)
                embedding_list.append(emb)
                embedding_id_map[(token, query_id)] = self.embedding_id_counter
                self.embedding_id_counter += 1
            
            # Free memory as we go
            embeddings[query_id] = None
        
        # Clear embeddings completely
        del embeddings
        
        # Convert embedding list to numpy array for efficiency
        embedding_array = np.array(embedding_list, dtype=np.float32)
        del embedding_list  # Free the list
        print(f"   ✓ Embedding array shape: {embedding_array.shape}, size: {embedding_array.nbytes / 1024 / 1024:.1f} MB")

        # Step 5: Build base table using IDs only
        print("🏗️  Building base table...")
        for query_id, sentence_pairs in tqdm(enumerate(pairs), total=len(pairs), desc="Step 5: Building base table", unit="query"):
            for center, context in sentence_pairs:
                if (center, query_id) not in embedding_id_map:
                    continue

                base_entry = {
                    'id': self.next_id,
                    'center_word_id': vocab_dict[center],
                    'center_pos': token_position_map[(center, query_id)],
                    'context_word_id': vocab_dict[context],
                    'embedding_id': embedding_id_map[(center, query_id)],
                    'sql_query_id': query_id
                }
                base_list.append(base_entry)
                self.next_id += 1

        # Convert to pandas DataFrames
        vocab_table = pd.DataFrame(list(vocab_dict.items()), columns=['word', 'id'])
        query_table = pd.DataFrame(query_list)
        
        # Create embedding table from numpy array
        embedding_table = pd.DataFrame({
            'id': range(len(embedding_array)),
            'embedding': list(embedding_array)
        })
        del embedding_array  # Free memory
        
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
