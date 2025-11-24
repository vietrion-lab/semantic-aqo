from typing import List
from tqdm import tqdm
from sensate.pipeline.preprocessing.bert_extractor import BERTExtractor

class PairGenerator:
    # Generate center-context pairs for Word2Vec training
    def __init__(self, window_size: int = 2):
        self.window_size = window_size

    def _get_context_indices(self, tokens: List[str], center_idx: int) -> List[int]:
        """Return indices of context tokens within the configured window."""
        left_start = max(0, center_idx - self.window_size)
        right_end = min(len(tokens), center_idx + self.window_size + 1)
        return list(range(left_start, center_idx)) + list(range(center_idx + 1, right_end))
 
    def generate_center_context_pair(self, tokens: List[str]) -> List[tuple]:
        """Generate (center_idx, context_idx) pairs for a sentence."""
        pairs = []
        for i in range(len(tokens)):
            for ctx_idx in self._get_context_indices(tokens, i):
                pairs.append((i, ctx_idx))
        return pairs


class BERTEmbeddingGenerator:
    def __init__(self, foundation_model_name: str = None, ipca=None, sentences_per_chunk: int = 32, mask_batch_size: int = 128):
        self.extractor = BERTExtractor(model_name=foundation_model_name, ipca=ipca)
        self.sentences_per_chunk = sentences_per_chunk
        self.mask_batch_size = mask_batch_size

    def iter_embeddings(self, corpus: List[List[str]], sentences_per_chunk: int = None):
        """Yield embeddings per sentence in small chunks to control memory."""
        if sentences_per_chunk is None:
            sentences_per_chunk = self.sentences_per_chunk
        total = len(corpus)
        for start in tqdm(range(0, total, sentences_per_chunk), desc="Generating BERT embeddings", unit="sentence"):
            chunk = corpus[start:start + sentences_per_chunk]
            if not chunk:
                continue
            masked_sentences = []
            sentence_token_map = []
            for local_idx, sentence in enumerate(chunk):
                for token_idx in range(len(sentence)):
                    masked_sentence = sentence.copy()
                    masked_sentence[token_idx] = "<mask>"
                    masked_sentences.append(masked_sentence)
                    sentence_token_map.append((local_idx, token_idx))
            if not masked_sentences:
                for _ in chunk:
                    yield []
                continue
            batch_embeddings = []
            for i in range(0, len(masked_sentences), self.mask_batch_size):
                batch = masked_sentences[i:i + self.mask_batch_size]
                batch_embeddings.extend(self.extractor.batch_extract(batch))
            sentence_embeddings = [[None] * len(sentence) for sentence in chunk]
            for (local_idx, token_idx), embedding in zip(sentence_token_map, batch_embeddings):
                sentence_embeddings[local_idx][token_idx] = embedding
            for embeddings in sentence_embeddings:
                yield embeddings

    def process_sentence(self, sentence: List[str]) -> List:
        """Convenience method to process a single sentence."""
        return next(self.iter_embeddings([sentence], sentences_per_chunk=1))

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