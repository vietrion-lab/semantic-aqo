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

        # Step 2: Build vocabulary
        print("📚 Building vocabulary...")
        unique_words = set(word for sentence in corpus for word in sentence)
        vocab_dict = {}
        for word in tqdm(unique_words, desc="Step 2: Building vocab", unit="word"):
            vocab_dict[word] = self.word_id_counter
            self.word_id_counter += 1

        # Step 3: Build query table (lightweight, just metadata)
        print("📝 Building query table...")
        query_data = {
            'id': list(range(len(corpus))),
            'sql_query': [" ".join(s) for s in tqdm(corpus, desc="Step 3: Building queries", unit="query")],
            'sql_length': [len(s) for s in corpus]
        }
        query_table = pd.DataFrame(query_data)
        self.query_id_counter = len(corpus)
        del query_data  # Free immediately

        # Step 4+5: Build embedding and base tables together (single pass, memory efficient)
        print("🔢 Building embedding and base tables together...")
        
        # Get embedding dimension from first available embedding
        first_emb = None
        for emb_dict in embeddings:
            if emb_dict:
                first_emb = next(iter(emb_dict.values()))
                break
        
        if first_emb is None:
            raise ValueError("No embeddings found in corpus")
        
        # Get dimension - handle both numpy arrays and lists
        if isinstance(first_emb, np.ndarray):
            emb_dim = first_emb.shape[0]
        else:
            emb_dim = len(first_emb)
        
        # Use pre-sized numpy arrays instead of lists
        total_embeddings = sum(len(emb_dict) for emb_dict in embeddings)
        embedding_array = np.empty((total_embeddings, emb_dim), dtype=np.float32)
        
        # Build base table in chunks using numpy arrays
        chunk_size = 100000
        base_chunks = []
        
        # Temporary arrays for current chunk
        chunk_ids = np.empty(chunk_size, dtype=np.int32)
        chunk_center_word_ids = np.empty(chunk_size, dtype=np.int32)
        chunk_center_pos = np.empty(chunk_size, dtype=np.int32)
        chunk_context_word_ids = np.empty(chunk_size, dtype=np.int32)
        chunk_embedding_ids = np.empty(chunk_size, dtype=np.int32)
        chunk_query_ids = np.empty(chunk_size, dtype=np.int32)
        chunk_idx = 0
        
        embedding_idx = 0
        
        for query_id in tqdm(range(len(pairs)), desc="Step 4+5: Building tables", unit="query"):
            emb_dict = embeddings[query_id]
            sentence_pairs = pairs[query_id]
            sentence = corpus[query_id]
            
            # Build token position map for this query only (not global)
            token_positions = {token: pos for pos, token in enumerate(sentence)}
            
            # Build embedding map for this query only + store embeddings
            local_emb_map = {}
            for token, emb in emb_dict.items():
                embedding_array[embedding_idx] = emb
                local_emb_map[token] = embedding_idx
                embedding_idx += 1
            
            # Build base entries for this query using numpy arrays
            for center, context in sentence_pairs:
                if center not in local_emb_map:
                    continue

                chunk_ids[chunk_idx] = self.next_id
                chunk_center_word_ids[chunk_idx] = vocab_dict[center]
                chunk_center_pos[chunk_idx] = token_positions[center]
                chunk_context_word_ids[chunk_idx] = vocab_dict[context]
                chunk_embedding_ids[chunk_idx] = local_emb_map[center]
                chunk_query_ids[chunk_idx] = query_id
                
                chunk_idx += 1
                self.next_id += 1
                
                # When chunk is full, convert to DataFrame and reset
                if chunk_idx >= chunk_size:
                    base_chunks.append(pd.DataFrame({
                        'id': chunk_ids.copy(),
                        'center_word_id': chunk_center_word_ids.copy(),
                        'center_pos': chunk_center_pos.copy(),
                        'context_word_id': chunk_context_word_ids.copy(),
                        'embedding_id': chunk_embedding_ids.copy(),
                        'sql_query_id': chunk_query_ids.copy()
                    }))
                    chunk_idx = 0
            
            # Free memory for this query immediately
            embeddings[query_id] = None
            pairs[query_id] = None
        
        # Clear large structures
        del embeddings, pairs
        
        # Add remaining entries
        if chunk_idx > 0:
            base_chunks.append(pd.DataFrame({
                'id': chunk_ids[:chunk_idx].copy(),
                'center_word_id': chunk_center_word_ids[:chunk_idx].copy(),
                'center_pos': chunk_center_pos[:chunk_idx].copy(),
                'context_word_id': chunk_context_word_ids[:chunk_idx].copy(),
                'embedding_id': chunk_embedding_ids[:chunk_idx].copy(),
                'sql_query_id': chunk_query_ids[:chunk_idx].copy()
            }))
        
        # Free chunk arrays
        del chunk_ids, chunk_center_word_ids, chunk_center_pos, chunk_context_word_ids, chunk_embedding_ids, chunk_query_ids
        
        # Concatenate all chunks efficiently
        print(f"   ✓ Concatenating {len(base_chunks)} chunks...")
        base_table = pd.concat(base_chunks, ignore_index=True) if base_chunks else pd.DataFrame()
        del base_chunks
        
        print(f"   ✓ Embedding array shape: {embedding_array.shape}, size: {embedding_array.nbytes / 1024 / 1024:.1f} MB")

        # Convert to pandas DataFrames
        vocab_table = pd.DataFrame(list(vocab_dict.items()), columns=['word', 'id'])
        
        # Create embedding table from numpy array
        embedding_table = pd.DataFrame({
            'id': range(len(embedding_array)),
            'embedding': list(embedding_array)
        })
        del embedding_array  # Free memory

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
