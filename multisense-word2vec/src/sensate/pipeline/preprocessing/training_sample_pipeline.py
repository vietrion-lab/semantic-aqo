from tabulate import tabulate
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans

from sensate.pipeline.preprocessing.corpus_pipeline import PairGenerator, BERTEmbeddingGenerator
from sensate.schema.config_schema import BaseTableEntry


class TrainingSampleGenerator:
    def __init__(self, window_size: int = 2, num_senses: int = 3):
        self.pair_generator = PairGenerator(window_size=window_size)
        self.embedding_generator = BERTEmbeddingGenerator()
        self.num_senses = num_senses
        self.next_id = 0
        self.word_id_counter = 0
        self.query_id_counter = 0
        self.embedding_id_counter = 0

    def __call__(self, corpus: List[List[str]]) -> tuple[Dict, Dict, Dict, List]:
        """
        4 tables:
        - vocab_table
        - query_table
        - embedding_table (includes sense & mixture embeddings)
        - base_table (each pair with mixture embedding id)
        """
        # Step 1: Generate pairs (center-context) and contextual BERT embeddings
        pairs = self.pair_generator(corpus)
        embeddings = self.embedding_generator(corpus)  # [{token: np.array(768)}, ...]

        vocab_table, query_table, embedding_table, base_table = {}, {}, {}, []

        # Step 2: Build vocabulary
        unique_words = set(word for sentence in corpus for word in sentence)
        for word in unique_words:
            vocab_table[word] = self.word_id_counter
            self.word_id_counter += 1

        # Step 3: Build query table
        for i, sentence in enumerate(corpus):
            query_table[i] = " ".join(sentence)
            self.query_id_counter += 1

        # Step 4: Collect all BERT embeddings per token
        word_to_vectors = {word: [] for word in vocab_table.keys()}
        for emb_dict in embeddings:
            for token, emb in emb_dict.items():
                if token in word_to_vectors:
                    word_to_vectors[token].append(np.array(emb))

        # Step 5️: Cluster each word’s embeddings into K senses
        word_senses = {}
        for word, vecs in word_to_vectors.items():
            if len(vecs) >= self.num_senses:
                km = KMeans(n_clusters=self.num_senses, random_state=42)
                km.fit(vecs)
                word_senses[word] = km.cluster_centers_
            elif len(vecs) > 0:
                # Nếu số ngữ cảnh < K → nhân bản vector đầu để đủ K cụm
                word_senses[word] = np.tile(vecs[0], (self.num_senses, 1))
            else:
                # fallback nếu không có embedding
                word_senses[word] = np.zeros((self.num_senses, 768))

        # Step 6️: Add all sense vectors to embedding_table
        for word, sense_vecs in word_senses.items():
            for s_i, vec in enumerate(sense_vecs):
                embedding_table[self.embedding_id_counter] = vec.tolist()
                self.embedding_id_counter += 1

        # Step 7️: Build Base Table with mixture embeddings per occurrence
        for i, sentence_pairs in enumerate(pairs):
            query_id = i
            emb_dict = embeddings[i]
            for center, context in sentence_pairs:
                if center not in word_senses or center not in emb_dict:
                    continue

                # Tính mixture embedding theo độ tương đồng với 3 sense
                current_emb = np.array(emb_dict[center])
                sense_vecs = np.array(word_senses[center])
                sims = sense_vecs @ current_emb / (
                    np.linalg.norm(sense_vecs, axis=1) * np.linalg.norm(current_emb) + 1e-8
                )
                weights = np.exp(sims) / np.sum(np.exp(sims))
                mixture_emb = np.sum(weights[:, None] * sense_vecs, axis=0)

                # Lưu mixture embedding vào embedding_table
                embedding_table[self.embedding_id_counter] = mixture_emb.tolist()
                embedding_id = self.embedding_id_counter
                self.embedding_id_counter += 1

                base_entry = BaseTableEntry(
                    id=self.next_id,
                    center_word=center,
                    context_word=context,
                    sql_query=query_table[query_id],
                    embedding_id=embedding_id
                )
                base_table.append(base_entry)
                self.next_id += 1

        return vocab_table, query_table, embedding_table, base_table
# def print_pretty_tables(vocab_table: Dict, query_table: Dict, embedding_table: Dict, base_table: List):
#     # Print Vocabulary Table
#     print("\n Vocabulary Table:")
#     vocab_data = [[word, id] for word, id in vocab_table.items()]
#     print(tabulate(vocab_data, headers=["Word", "ID"], tablefmt="fancy_grid"))
#
#     # Print Query Table
#     print("\n Query Table:")
#     query_data = [[qid, query] for qid, query in query_table.items()]
#     print(tabulate(query_data, headers=["Query ID", "SQL Query"], tablefmt="fancy_grid"))
#
#     # Print Embedding Table
#     print("\n Embedding Table (first 5 dims):")
#     embedding_data = [[eid, emb[:5]] for eid, emb in embedding_table.items()]
#     print(tabulate(embedding_data, headers=["Embedding ID", "Embedding"], tablefmt="fancy_grid"))
#
#     # Print Base Table
#     print("\n Base Table:")
#     base_data = [[entry.id, entry.center_word, entry.context_word, entry.sql_query, entry.embedding_id]
#                  for entry in base_table]
#     print(tabulate(base_data, headers=["ID", "Center", "Context", "SQL Query", "Embedding ID"], tablefmt="fancy_grid"))

#
# if __name__ == "__main__":
#     sample_corpus = [
#         ["SELECT", "<TAB>", "WHERE", "<COL>", "=", "<STR>"],
#         ["SELECT", "<COL>", "FROM", "<TAB>", "JOIN", "<TAB>", "ON", "<TAB>", ".", "<COL>", "=", "<TAB>", ".", "<COL>"]
#     ]
#
#     training_sample_generator = TrainingSampleGenerator(window_size=2)
#     vocab_table, query_table, embedding_table, base_table = training_sample_generator(corpus=sample_corpus)
#
#     # Print tables
#     print_pretty_tables(vocab_table, query_table, embedding_table, base_table)
