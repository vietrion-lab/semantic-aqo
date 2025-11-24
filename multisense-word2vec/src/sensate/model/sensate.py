import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from sensate.pipeline.training.initialization import (
    SenseEmbeddingsInitializer
)
from sensate.model.gating_network_layer import GatingNetworkLayer
from sensate.pipeline.training.embedding_storage import optimize_embedding_table



class Sensate(nn.Module):
    """
    Sensate Model combining BERT-based sense embeddings with output and projection matrices.
    """
    def __init__(
        self,
        base_table: pd.DataFrame,
        vocab_table: pd.DataFrame,
        embedding_table: pd.DataFrame,
        query_table: pd.DataFrame,
        num_senses: int,
        embedding_dim: int
    ):
        super(Sensate, self).__init__()
        self.base_table = base_table
        self.vocab_table = vocab_table
        # Optimize embedding table storage to save memory
        print("🔧 Optimizing embedding table for memory efficiency...")
        self.embedding_table = optimize_embedding_table(
            embedding_table,
            cache_size=10000,
            use_float16=True  # Save 50% memory with float16
        )
        self.query_table = query_table
        self.num_senses = num_senses
        self.embedding_dim = embedding_dim
        self.gating_network_layer = GatingNetworkLayer(sigma=0.001, d=embedding_dim)
        
        # Initialize
        # Convert embedding_table back to DataFrame for initialization (only temporarily)
        embedding_df = self.embedding_table.to_dataframe()
        sense_embeddings, updated_embedding_df = SenseEmbeddingsInitializer(
            base_table=base_table, 
            vocab=vocab_table, 
            embedding=embedding_df,
            embedding_dim=embedding_dim,
            num_senses=num_senses
        )()
        
        # Re-optimize the updated embedding table
        del self.embedding_table  # Free old storage
        self.embedding_table = optimize_embedding_table(
            updated_embedding_df,
            cache_size=10000,
            use_float16=True
        )
        
        if len(updated_embedding_df) > 0:
            print(f"    Each Embedding Shape: {len(updated_embedding_df['embedding'].iloc[0])} dims")
            # Check if all embedding dim not match
            assert all(len(emb) == embedding_dim for emb in updated_embedding_df['embedding']), \
                "All embeddings must have the correct embedding_dim"
        

        print(f"Sense Embeddings Parameter: {sense_embeddings.shape}")
        print("=" * 20)

        output_embeddings_tensor = torch.randn(vocab_table.shape[0], embedding_dim) * 0.01
        
        # Register as trainable parameters
        self.sense_embeddings = nn.Parameter(sense_embeddings)          # [V, K, D]
        self.output_embeddings = nn.Parameter(output_embeddings_tensor) # [V, D]
        print(f"Output Embeddings: {self.output_embeddings.shape}")
        print("=" * 20)

    def forward(self, 
                center_pos,            # [B] - center position in each query
                context_ids,           # [B] - context word ids (single context per sample)
                query_token_ids,       # [B, T] - token ids for each position in the query
                bert_embeddings        # [B, D] - BERT embeddings
    ) -> torch.Tensor:
        batch_size = query_token_ids.shape[0]
        
        # Use advanced indexing instead of gather for faster performance on A100
        center_word_ids = query_token_ids[torch.arange(batch_size, device=query_token_ids.device), center_pos]  # [B]
        center_sense_embeddings = self.sense_embeddings[center_word_ids]  # [B, K, D]
        
        all_query_sense_embeddings = self.sense_embeddings[query_token_ids]  # [B, T, K, D]
        
        # Vectorized context extraction using masking
        T = query_token_ids.shape[1]
        pos_indices = torch.arange(T, device=query_token_ids.device).unsqueeze(0).expand(batch_size, -1)  # [B, T]
        context_mask = pos_indices != center_pos.unsqueeze(1)  # [B, T]
        
        # More efficient reshaping
        context_mask_expanded = context_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        masked_embeddings = all_query_sense_embeddings * context_mask_expanded  # Broadcasting
        context_sense_embeddings = masked_embeddings[context_mask].view(batch_size, T-1, self.num_senses, self.embedding_dim)  # [B, T-1, K, D]
        
        max_pooled_embedding, gating_probs = self.gating_network_layer(
            center_pos=center_pos,
            query_token_ids=query_token_ids,
            center_sense_embeddings=center_sense_embeddings,
            context_sense_embeddings=context_sense_embeddings
        ) # shape: (batch_size, embedding_dim), (batch_size, num_senses)
        
        # Optimized loss computations using fused operations
        # Word2vec SG naive softmax loss - use @ for faster matmul on A100
        logits = max_pooled_embedding @ self.output_embeddings.T  # [B, V]
        L_w2v = F.cross_entropy(logits, context_ids, reduction='mean')

        # Distillation loss - fused MSE
        L_distill = F.mse_loss(max_pooled_embedding, bert_embeddings, reduction='mean')
        
        # Orthogonality loss - optimized with einsum for A100 tensor cores
        normalized_embeddings = F.normalize(center_sense_embeddings, p=2, dim=-1)  # [B, K, D]
        similarity_matrix = torch.bmm(normalized_embeddings, normalized_embeddings.transpose(1, 2))  # [B, K, K]
        # Only upper triangle, excluding diagonal
        triu_mask = torch.triu(torch.ones_like(similarity_matrix[0]), diagonal=1).bool()
        L_orth = (similarity_matrix[:, triu_mask].pow(2)).mean()
        
        # Entropy loss - numerically stable
        L_ent = -(gating_probs * torch.log(gating_probs.clamp(min=1e-12))).sum(dim=-1).mean()
        
        # L2 regularization - use torch.norm for better performance
        L2_reg = sum(torch.norm(p, p=2)**2 for p in self.parameters())
        
        # Combine losses
        total_loss = L_w2v + 0.3 * L_distill + 0.1 * L_orth + 0.01 * L_ent + 0.001 * L2_reg

        return total_loss