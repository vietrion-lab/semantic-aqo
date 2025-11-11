import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import pandas as pd
import os
from sklearn.cluster import KMeans
from tqdm import tqdm
from sklearn.metrics import classification_report

from sensate.pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline

class Evaluator(nn.Module):
    """
    Evaluator Module to compute similarity scores between word embeddings.
    """
    def __init__(self, evaluation_datasets_path='../evaluation_datasets'):
        super(Evaluator, self).__init__()
        # Use absolute path or allow configuration
        self.bombay_df = pd.read_csv(os.path.join(evaluation_datasets_path, 'bombay_queries.csv'), sep='\t')
        self.googleplus_df = pd.read_csv(os.path.join(evaluation_datasets_path, 'googleplus_queries.csv'), sep='\t')
        self.ub_df = pd.read_csv(os.path.join(evaluation_datasets_path, 'ub_queries.csv'), sep='\t')
        print(f"📊 Loaded {len(self.bombay_df)} Bombay, {len(self.googleplus_df)} GooglePlus, {len(self.ub_df)} UB queries")
        self.pipeline = PreprocessingPipeline()
        
        # Cache preprocessed queries to avoid reprocessing every evaluation
        self._preprocessed_cache = {
            'bombay': None,
            'googleplus': None,
            'ub': None
        }

    def _gating_network_inference(self, center_id: int, center_pos: int, query_ids: list, embedding_table: torch.Tensor, sigma: float = 0.001):
        """
        Perform gating network inference to get sense-aware embedding for center word.
        Matches the logic in gating_network_layer.py
        """
        device = embedding_table.device  # Get device from embedding_table
        
        # Get embeddings
        center_sense_embeddings = embedding_table[center_id]  # [K, D]
        
        # Get context (exclude center position)
        context_ids = [query_ids[i] for i in range(len(query_ids)) if i != center_pos]
        context_sense_embeddings = embedding_table[context_ids]  # [T-1, K, D]
        
        num_senses = center_sense_embeddings.shape[0]
        embedding_dim = center_sense_embeddings.shape[1]
        
        # Mean over sense dimension: [T-1, K, D] -> [T-1, D]
        v = context_sense_embeddings.mean(dim=1)  # [T-1, D]
        
        # Attention scores: [K, D] @ [D, T-1] -> [K, T-1]
        a = torch.matmul(center_sense_embeddings, v.T) / math.sqrt(embedding_dim)
        a = a.T  # [T-1, K]
        
        # Positional weights - create on same device
        context_positions = torch.tensor([i for i in range(len(query_ids)) if i != center_pos], 
                                        dtype=torch.float32, device=device)
        w = torch.exp(- torch.abs(context_positions - center_pos) / sigma)  # [T-1]
        
        # Weighted attention: [T-1, K] + [T-1, 1]
        alpha = torch.softmax(a + w.unsqueeze(-1).log(), dim=0)  # [T-1, K]
        
        # Weighted context: [K, T-1] @ [T-1, D] -> [K, D]
        weighted_context = torch.matmul(alpha.T, v)  # [K, D]
        
        # Cosine similarity between center senses and weighted context
        s = F.cosine_similarity(center_sense_embeddings, weighted_context, dim=-1)  # [K]
        q = torch.softmax(s, dim=-1)  # [K]
        
        # Select best sense
        idx = q.argmax(dim=-1)
        sense_aware_embedding = center_sense_embeddings[idx]  # [D]

        return sense_aware_embedding

    def _infer(self, target: str, embedding_table: torch.Tensor, vocab_table: dict):
        if target == 'bombay':
            df = self.bombay_df
        elif target == 'googleplus':
            df = self.googleplus_df
        else:
            df = self.ub_df

        # Use cached preprocessing results (silent mode - no progress bars)
        if self._preprocessed_cache[target] is None:
            self._preprocessed_cache[target] = self.pipeline(df['query'].tolist(), verbose=False)
        queries = self._preprocessed_cache[target]
        
        query_embeddings = []
        
        # Disable progress bar for cleaner logs
        for query in queries:
            embeddings = []
            
            # Map positions: only include tokens that exist in vocab
            valid_positions = []
            valid_ids = []
            for pos, token in enumerate(query):
                if token in vocab_table:
                    valid_positions.append(pos)
                    valid_ids.append(vocab_table[token])
            
            # Process each valid token
            for idx, (pos, token) in enumerate(zip(valid_positions, [query[p] for p in valid_positions])):
                center_id = vocab_table[token]
                
                # Get sense-aware embedding for center word
                # Use idx (position in valid_ids) instead of pos (position in original query)
                embeddings.append(self._gating_network_inference(
                    center_id,
                    idx,  # Position in the filtered list
                    valid_ids,  # Only IDs that exist in vocab
                    embedding_table
                ))
            
            # Average over all center words in the query
            if len(embeddings) > 0:
                query_emb = torch.stack(embeddings).mean(dim=0)
                query_embeddings.append(query_emb)

        return query_embeddings

    def forward(self, embedding_table: torch.Tensor, vocab_table: dict):
        bombay_embeddings  = self._infer('bombay', embedding_table, vocab_table)
        googleplus_embeddings = self._infer('googleplus', embedding_table, vocab_table)
        ub_embeddings = self._infer('ub', embedding_table, vocab_table)

        # Convert to numpy for KMeans
        bombay_np = torch.stack(bombay_embeddings).cpu().numpy()
        googleplus_np = torch.stack(googleplus_embeddings).cpu().numpy()
        ub_np = torch.stack(ub_embeddings).cpu().numpy()

        bombay_cluster = KMeans(n_clusters=14, random_state=0)
        googleplus_cluster = KMeans(n_clusters=8, random_state=0)
        ub_cluster = KMeans(n_clusters=2, random_state=0)

        bombay_results = bombay_cluster.fit_predict(bombay_np)
        googleplus_results = googleplus_cluster.fit_predict(googleplus_np)
        ub_results = ub_cluster.fit_predict(ub_np)

        bombay_report = classification_report(self.bombay_df['label'].values[:len(bombay_results)], bombay_results)
        googleplus_report = classification_report(self.googleplus_df['label'].values[:len(googleplus_results)], googleplus_results)
        ub_report = classification_report(self.ub_df['label'].values[:len(ub_results)], ub_results)

        return {
            'bombay_report': bombay_report,
            'googleplus_report': googleplus_report,
            'ub_report': ub_report
        }