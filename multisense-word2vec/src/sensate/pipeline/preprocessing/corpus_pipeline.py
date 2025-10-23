from typing import List, Dict
from tqdm import tqdm
from sensate.pipeline.preprocessing.bert_extractor import BERTExtractor

class PairGenerator:
    # Generate center-context pairs for Word2Vec training
    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def _get_contexts(self, tokens: List[str], center_idx: int) -> List[str]:
        # Purpose: Extract left and right contexts within window size for a given center
        left_start = max(0, center_idx - self.window_size)
        left_contexts = tokens[left_start:center_idx]
        right_end = min(len(tokens), center_idx + self.window_size + 1)
        right_contexts = tokens[center_idx + 1:right_end]
        return left_contexts + right_contexts

    def generate_center_context_pair(self, tokens: List[str]) -> List[List[str]]:
        # Purpose: Generate pairs [center, context] for a single tokenized sentence
        pairs = []
        for i in range(len(tokens)):
            center = tokens[i]
            contexts = self._get_contexts(tokens, i)
            for context in contexts:
                pairs.append([center, context])
        return pairs

    def __call__(self, corpus: list) -> list:
        # Purpose: Process a corpus to generate pairs for all sentences
        return [self.generate_center_context_pair(sentence) for sentence in tqdm(corpus, desc="Generating training samples", unit="sentence")]


class BERTEmbeddingGenerator:
    def __init__(self):
        self.extractor = BERTExtractor()

    def __call__(self, corpus: List[List[str]]) -> List[Dict[str, list]]:
        """
        For each sentence in corpus:
            - Mask each token one by one.
            - Use BERTExtractor to extract embedding for the masked token.
        Output:
            List[Dict[token, embedding_vector]]
        """
        embeddings_list = []

        for sentence in tqdm(corpus, desc="Generating BERT embedding for queries", unit="query"):
            embeddings_dict = {}
            for i, token in enumerate(sentence):
                masked_sentence = sentence.copy()
                masked_sentence[i] = "[MASK]"
                embedding = self.extractor(masked_sentence)
                embeddings_dict[token] = embedding

            embeddings_list.append(embeddings_dict)

        return embeddings_list

# Test code with multiple sample data
# if __name__ == "__main__":
#     # Multiple sample inputs based on provided figures, with [MASK] added
#     sample_corpus = [
#         ["SELECT", "<TAB>", "WHERE", "<COL>", "=", "<STR>"],
#         ["SELECT", "<COL>", "FROM", "<TAB>", "JOIN", "<TAB>", "ON", "<TAB>", ".", "<COL>", "=", "<TAB>", ".", "<COL>"]
#     ]
#
#     # Initialize PairGenerator with default window_size=2 and BERTEmbeddingGenerator
#     pair_generator = PairGenerator(window_size=2)
#     embedding_generator = BERTEmbeddingGenerator()
#
#     # Generate pairs with default window_size
#     print("Generated center-context pairs (window_size=2):")
#     pairs = pair_generator(corpus=sample_corpus)
#     embeddings = embedding_generator(corpus=sample_corpus)
#
#     for i, sentence_pairs in enumerate(pairs):
#         print(f"\nSentence {i + 1} Center-Context pairs:")
#         for pair in sentence_pairs:
#             print(pair)
#
#     for i, embedding_dict in enumerate(embeddings):
#         print(f"\nSentence {i + 1} BERT Embedding dict:")
#         for token, emb in embedding_dict.items():
#             print(f"  {token}: {len(emb)}-dim vector")