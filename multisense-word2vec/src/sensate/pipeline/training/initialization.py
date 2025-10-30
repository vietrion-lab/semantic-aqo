from tqdm import tqdm
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import IncrementalPCA

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
        
    def _fit_ipca(self) -> IncrementalPCA:
        ipca = IncrementalPCA(n_components=self.embedding_dim, batch_size=3000)
        reduction_embeddings = []
        
        for _, row in tqdm(self.vocab.iterrows(), desc="Fitting IPCA", unit="word"):
            id = row['id']
            embedding_ids = self.base_table[self.base_table['center_word_id'] == id]['embedding_id'].unique()
            embeddings = self.embedding[self.embedding['id'].isin(embedding_ids)]
            for emb in embeddings.itertuples():
                reduction_embeddings.append(emb.embedding)
            
        print(f"Fitting IncrementalPCA with {len(reduction_embeddings)} embeddings.")
        
        batch_size = 3000
        for i in tqdm(range(0, len(reduction_embeddings), batch_size), desc="Fitting IPCA", unit="batch"):
            batch_embeddings = reduction_embeddings[i:min(i+batch_size, len(reduction_embeddings))]
            ipca.partial_fit(batch_embeddings)
        retained_var = ipca.explained_variance_ratio_.sum()
        print(f"✓ Retained variance {retained_var:.4f}")
        
        return ipca
    
    def _cluster_embeddings(self, ipca: IncrementalPCA):
        # Initialize sense embeddings based on base_table and vocab
        print(f"Clustering {self.num_senses} Sense Embedding for {len(self.vocab)} words.")
        
        sense_emb_table = {}
        
        for _, row in tqdm(self.vocab.iterrows(), desc="Initializing Sense Embeddings", unit="word"):
            word = row['word']
            id = row['id']
            embedding_ids = self.base_table[self.base_table['center_word_id'] == id]['embedding_id'].unique()
            embeddings = self.embedding[self.embedding['id'].isin(embedding_ids)]

            assert len(embeddings) >= self.num_senses, \
                f"  Not enough embeddings ({len(embeddings)}) for word '{word}' to form {self.num_senses} senses."
            
            features = np.array(embeddings['embedding'].tolist())
            transformed_features = ipca.transform(features)
            
            # Update each transformed embedding in the correct position
            # Important: We need to match the order of embeddings DataFrame, not embedding_ids
            for idx, (emb_idx, emb_row) in enumerate(embeddings.iterrows()):
                self.embedding.at[emb_idx, 'embedding'] = transformed_features[idx]
        
            kmeans = KMeans(n_clusters=self.num_senses, random_state=0)
            kmeans.fit(transformed_features)
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
        ipca = self._fit_ipca()
        sense_df = self._cluster_embeddings(ipca)
        
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