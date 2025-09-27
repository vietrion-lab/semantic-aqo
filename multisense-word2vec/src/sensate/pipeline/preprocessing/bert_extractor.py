from typing import List

from sensate.factory.common import device


class BERTExtractor:
    # TODO: AQO-7 - Hint: You have to use GPU to enhance the processing speed
    def __init__(self):
        pass
    
    def __call__(self, tokens: List[str]) -> list:
        pass
    
# Usage
# extractor = BERTExtractor()
# embeddings = extractor(["SELECT", "[MASK]", "FROM", "<TAB>", "WHERE", "<COL>", "=", "<STR>", ";"])
# print(embeddings) # Must return a list with 768 elements