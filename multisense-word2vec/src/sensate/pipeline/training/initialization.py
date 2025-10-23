from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

class SenseEmbeddingsInitializer:
    def __init__(self, 
            base_table: pd.DataFrame, 
            vocab: pd.DataFrame, 
            embedding: pd.DataFrame,
            embedding_dim: int,
            num_senses: int
        ):
        assert base_table is not None, "base_table must be provided"
        assert vocab is not None, "vocab must be provided"
        assert embedding is not None, "embedding must be provided"
        assert embedding_dim is not None, "embedding_dim must be provided"
        assert num_senses is not None, "num_senses must be provided"
        
        self.base_table = base_table
        self.vocab = vocab
        self.embedding = embedding
        self.embedding_dim = embedding_dim
        self.num_senses = num_senses

    def __call__(self) -> pd.DataFrame:
        # Initialize sense embeddings based on base_table and vocab
        print(f"Clustering {self.num_senses} Sense Embedding for {len(self.vocab)} words.")
        
        sense_emb_table = {}
        
        for _, row in tqdm(self.vocab.iterrows(), desc="Initializing Sense Embeddings", unit="word"):
            word = row['word']
            id = row['id']
            embedding_ids = self.base_table[self.base_table['center_word_id'] == id]['embedding_id'].unique()
            embeddings = self.embedding[self.embedding['id'].isin(embedding_ids)]
            
            if len(embeddings) < self.num_senses:
                print(f"  Not enough embeddings ({len(embeddings)}) for word '{word}' to form {self.num_senses} senses. Skipping.")
                continue
            
            features = np.array(embeddings['embedding'].tolist())
        
            kmeans = KMeans(n_clusters=self.num_senses, random_state=0)
            kmeans.fit(features)
            centers = kmeans.cluster_centers_

            # Store each sense embedding for this word
            for sense_idx in range(self.num_senses):
                sense_emb_table[f"{word}_sense_{sense_idx}"] = centers[sense_idx]
        
        sense_df = pd.DataFrame(sense_emb_table)
        sense_records = []
        for word in sense_emb_table.keys():
            # Extract word and sense index from column name
            parts = word.rsplit('_sense_', 1)
            if len(parts) == 2:
                word_name = parts[0]
                sense_idx = int(parts[1])
                embedding = sense_emb_table[word]
                sense_records.append({
                    'word': word_name,
                    'sense_id': sense_idx,
                    'embedding': embedding.tolist()
                })
        
        sense_df = pd.DataFrame(sense_records)
        return sense_df
        
class OutputEmbeddingsInitializer:
    # TODO: AQO-11
    def __init__(self, vocab, config):
        self.vocab = vocab
        self.config = config

    def __call__(self):
        # Initialize output embeddings based on vocab
        # Placeholder for actual initialization logic
        return {"output_embeddings": "initialized"}
    
class ProjectionMatricesInitializer:
    # TODO: AQO-11
    def __init__(self):
        pass

    def __call__(self):
        # Initialize projection matrices based on vocab
        print(f"Initializing projection matrices")
        # Placeholder for actual initialization logic
        return {"projection_matrices": "initialized"}