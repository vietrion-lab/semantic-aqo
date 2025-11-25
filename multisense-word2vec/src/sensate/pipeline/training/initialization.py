from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
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
        
    def _cluster_embeddings(self):
        # Initialize sense embeddings based on base_table and vocab
        print(f"Clustering {self.num_senses} Sense Embedding for {len(self.vocab)} words.")
        
        sense_emb_table = {}
        
        for _, row in tqdm(self.vocab.iterrows(), desc="Initializing Sense Embeddings", unit="word"):
            word = row['word']
            id = row['id']
            embedding_ids = self.base_table[self.base_table['center_word_id'] == id]['embedding_id'].unique()
            embeddings = self.embedding[self.embedding['id'].isin(embedding_ids)]

            # Handle words with fewer embeddings than num_senses
            if len(embeddings) < self.num_senses:
                # Use available embeddings and duplicate to fill remaining senses
                features = np.array(embeddings['embedding'].tolist())
                
                # If only 1 embedding, duplicate it for all senses
                if len(features) == 1:
                    centers = np.tile(features[0], (self.num_senses, 1))
                else:
                    # Use available embeddings and add noise for remaining senses
                    centers = []
                    for i in range(self.num_senses):
                        if i < len(features):
                            centers.append(features[i])
                        else:
                            # Add noise to last embedding for additional senses
                            noise = np.random.randn(features.shape[1]) * 0.01
                            centers.append(features[-1] + noise)
                    centers = np.array(centers)
            else:
                # Normal clustering with enough embeddings
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
                    'word_id': self.vocab[self.vocab['word'] == word_name]['id'].values[0],
                    'sense_id': sense_idx,
                    'embedding': embedding.tolist()
                })
        
        sense_df = pd.DataFrame(sense_records)

        return sense_df

    def __call__(self) -> torch.Tensor:
        sense_df = self._cluster_embeddings()
        
        # Convert sense_df to tensor [n_vocab, n_sense, dim]
        # Handle non-incremental word_id order by using max word_id + 1 as size
        n_sense = self.num_senses
        dim = self.embedding_dim
        
        # Get the maximum word_id to determine tensor size
        max_word_id = sense_df['word_id'].max()
        n_vocab = max_word_id + 1  # Size needs to accommodate the largest index
        
        sense_tensor = torch.zeros(n_vocab, n_sense, dim)
        for _, row in sense_df.iterrows():
            word_id = int(row['word_id'])
            sense_id = int(row['sense_id'])
            embedding = torch.tensor(row['embedding'], dtype=torch.float32)
            sense_tensor[word_id, sense_id, :] = embedding
        
        return sense_tensor, self.embedding