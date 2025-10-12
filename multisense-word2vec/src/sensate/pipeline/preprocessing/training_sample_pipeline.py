from tabulate import tabulate
from typing import List, Dict
from sensate.pipeline.preprocessing.corpus_pipeline import PairGenerator, BERTEmbeddingGenerator
from sensate.schema.config_schema import BaseTableEntry


class TrainingSampleGenerator:
    def __init__(self, window_size: int = 2):
        self.pair_generator = PairGenerator(window_size=window_size)
        self.embedding_generator = BERTEmbeddingGenerator()
        self.next_id = 0
        self.word_id_counter = 0
        self.query_id_counter = 0
        self.embedding_id_counter = 0

    def __call__(self, corpus: List[List[str]]) -> tuple[Dict, Dict, Dict, List]:
        pairs = self.pair_generator(corpus)
        embeddings = self.embedding_generator(corpus)

        vocab_table, query_table, embedding_table, base_table = {}, {}, {}, []
        embedding_id_map = {}

        unique_words = set(word for sentence in corpus for word in sentence)
        for word in unique_words:
            vocab_table[word] = self.word_id_counter
            self.word_id_counter += 1

        for i, sentence in enumerate(corpus):
            query_table[i] = " ".join(sentence)
            self.query_id_counter += 1

        for query_id, emb_dict in enumerate(embeddings):
            for token, emb in emb_dict.items():
                embedding_table[self.embedding_id_counter] = emb
                embedding_id_map[(token, query_id)] = self.embedding_id_counter
                self.embedding_id_counter += 1

        for query_id, sentence_pairs in enumerate(pairs):
            for center, context in sentence_pairs:
                if (center, query_id) not in embedding_id_map:
                    continue

                base_entry = BaseTableEntry(
                    id=self.next_id,
                    center_word_id=vocab_table[center],
                    context_word_id=vocab_table[context],
                    embedding_id=embedding_id_map[(center, query_id)],
                    sql_query_id=query_id
                )
                base_table.append(base_entry)
                self.next_id += 1

        print(f"Vocab Table size: {len(vocab_table)}")
        print(f"Query Table size: {len(query_table)}")
        print(f"Embedding Table size: {len(embedding_table)}")
        print(f"Base Table size: {len(base_table)}")

        return vocab_table, query_table, embedding_table, base_table

#
# def print_pretty_tables(vocab_table: Dict, query_table: Dict, embedding_table: Dict, base_table: List):
#     print("\nVocabulary Table:")
#     vocab_data = [[word, id] for word, id in vocab_table.items()]
#     print(tabulate(vocab_data, headers=["Word", "ID"], tablefmt="fancy_grid"))
#
#     print("\nQuery Table:")
#     query_data = [[qid, query] for qid, query in query_table.items()]
#     print(tabulate(query_data, headers=["Query ID", "SQL Query"], tablefmt="fancy_grid"))
#
#     print("\nEmbedding Table (first 5 dims):")
#     embedding_data = [[eid, emb[:5]] for eid, emb in embedding_table.items()]
#     print(tabulate(embedding_data, headers=["Embedding ID", "Embedding"], tablefmt="fancy_grid"))
#
#     print("\nBase Table:")
#     base_data = [
#         [entry.id, entry.center_word_id, entry.context_word_id, entry.embedding_id, entry.sql_query_id]
#         for entry in base_table
#     ]
#     print(tabulate(base_data, headers=["ID", "Center ID", "Context ID", "Embedding ID", "SQL Query ID"], tablefmt="fancy_grid"))
#
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
#     print_pretty_tables(vocab_table, query_table, embedding_table, base_table)
