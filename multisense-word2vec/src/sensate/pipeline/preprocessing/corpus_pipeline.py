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
    def __init__(self, foundation_model_name: str = None, ipca=None):
        self.extractor = BERTExtractor(model_name=foundation_model_name, ipca=ipca)

    def __call__(self, corpus: List[List[str]]) -> List[Dict[str, list]]:
        """
        For each sentence in corpus:
            - Mask each token one by one.
            - Use BERTExtractor to extract embedding for the masked token.
        Output:
            List[Dict[token, embedding_vector]]
        
        Optimized with deduplication: process each unique (sentence_pattern, token_position) only once.
        """
        # Initialize result structure
        embeddings_list = [{} for _ in corpus]
        
        # Step 1: Deduplicate - find unique (sentence_pattern, position) pairs
        print("🔍 Deduplicating sentences for faster processing...")
        unique_masks = {}  # (sentence_tuple, position) -> list of (sentence_idx, token)
        
        for sentence_idx, sentence in enumerate(corpus):
            sentence_tuple = tuple(sentence)
            for token_idx, token in enumerate(sentence):
                key = (sentence_tuple, token_idx)
                if key not in unique_masks:
                    unique_masks[key] = []
                unique_masks[key].append((sentence_idx, token))
        
        print(f"   ✓ Reduced from {sum(len(s) for s in corpus):,} to {len(unique_masks):,} unique masks ({100*(1-len(unique_masks)/sum(len(s) for s in corpus)):.1f}% reduction)")
        
        # Step 2: Process only unique masks in large batches
        batch_size = 512  # A100 can handle large batches
        
        unique_items = list(unique_masks.items())
        total_batches = (len(unique_items) + batch_size - 1) // batch_size
        
        for batch_start in tqdm(range(0, len(unique_items), batch_size),
                               total=total_batches,
                               desc="Generating BERT embeddings",
                               unit="batch"):
            batch_end = min(batch_start + batch_size, len(unique_items))
            batch_items = unique_items[batch_start:batch_end]
            
            # Prepare masked sentences for this batch
            masked_sentences = []
            batch_metadata = []
            
            for (sentence_tuple, token_idx), occurrences in batch_items:
                sentence = list(sentence_tuple)
                masked_sentence = sentence.copy()
                masked_sentence[token_idx] = "<mask>"
                masked_sentences.append(masked_sentence)
                batch_metadata.append(occurrences)
            
            # Get embeddings for this batch
            batch_embeddings = self.extractor.batch_extract(masked_sentences)
            
            # Assign embeddings to ALL occurrences
            for embedding, occurrences in zip(batch_embeddings, batch_metadata):
                for sentence_idx, token in occurrences:
                    embeddings_list[sentence_idx][token] = embedding
            
            # Free memory
            del batch_embeddings, masked_sentences
        
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