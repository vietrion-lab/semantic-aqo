from typing import List


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
        return [self.generate_center_context_pair(sentence) for sentence in corpus]


class BERTEmbeddingGenerator:
    # TODO: AQO-8
    def __init__(self):
        pass

    def __call__(self, corpus: list) -> list:
        pass
    

# Test code with multiple sample data
# if __name__ == "__main__":
    # # Multiple sample inputs based on provided figures
    # sample_corpus = [
    #     ["SELECT", "<ALIAS_T1>", "<COL>", "SUM", "DESC"],
    #     ["FROM", "<TAB>", "WHERE", "<COL>", "=", '<STR>'],
    #     ["JOIN", "<TAB>", "<ALIAS_T1>", "<COL>", "AS", "<COL_OUT>"]
    # ]

    # # Initialize TrainingSampleGenerator with default window_size=2
    # pair_generator = PairGenerator(window_size=2)
    # embedding_generator = BERTEmbeddingGenerator()

    # # Generate pairs with default window_size
    # print("Generated center-context pairs (window_size=2):")
    # pairs = pair_generator(corpus=sample_corpus)
    # embeddings = embedding_generator(corpus=sample_corpus)
    
    # for i, pairs in enumerate(pairs):
    #     print(f"Sentence {i + 1}: {pairs}")

    # for i, embedding_dict in enumerate(embeddings):
    #     print(f"Sentence {i + 1} Embedding dict: {embedding_dict}")