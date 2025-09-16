class PairGenerator:
    def __init__(self, window_size: int):
        self.window_size = window_size

    def generate_center_context_pair(self, token_list: list) -> list:
        # TODO: AQO-6 task
        pass

    def __call__(self, corpus: list) -> list:
        pairs = []
        for token_list in corpus:
            pairs.extend(self.generate_center_context_pair(token_list))
        return pairs
    
# Usage example:
# generator = PairGenerator(window_size=2)
# pairs = generator(corpus=[["this", "is", "a", "sample"], ...])
# print(pairs)