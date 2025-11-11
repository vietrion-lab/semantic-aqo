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
        center_word_ids = query_token_ids[torch.arange(batch_size), center_pos]  # [B]
        center_sense_embeddings = self.sense_embeddings[center_word_ids, :, :]  # [B, K, D]
        
        all_query_sense_embeddings = self.sense_embeddings[query_token_ids, :, :]  # [B, T, K, D]
        context_sense_embeddings = []
        for b in range(batch_size):
            # Concatenate all positions except center_pos[b]
            contexts = torch.cat([
                all_query_sense_embeddings[b, :center_pos[b], :, :],
                all_query_sense_embeddings[b, center_pos[b]+1:, :, :]
            ], dim=0)  # [T-1, K, D]
            context_sense_embeddings.append(contexts)
        context_sense_embeddings = torch.stack(context_sense_embeddings, dim=0)  # [B, T-1, K, D]
        
        max_pooled_embedding, gating_probs = self.gating_network_layer(
            center_pos=center_pos,
            query_token_ids=query_token_ids,
            center_sense_embeddings=center_sense_embeddings,
            context_sense_embeddings=context_sense_embeddings
        ) # shape: (batch_size, embedding_dim), (batch_size, num_senses)
        
        # Word2vec SG naive softmax loss
        # context_ids shape: [B], reshape to [B, 1] for gather operation
        L_w2v = -torch.log_softmax(max_pooled_embedding @ self.output_embeddings.T, dim=-1).gather(1, context_ids.unsqueeze(1)).mean()

        # Distillation loss
        # Both should be embedding_dim (150) after IPCA transformation
        L_distill = ((max_pooled_embedding - bert_embeddings) ** 2).mean()
        
        # Orthogonality loss
        L_orth = torch.triu(
            (F.normalize(center_sense_embeddings, p=2, dim=-1) @ 
            F.normalize(center_sense_embeddings, p=2, dim=-1).transpose(1, 2)).pow(2),
            diagonal=1
        ).sum(dim=(-1, -2)).mean()
        
        # Entropy loss
        L_ent = -(gating_probs * (gating_probs + 1e-12).log()).sum(dim=-1).mean()
        
        # L2 regularization
        L2_reg = sum(p.pow(2).sum() for p in self.parameters())
        
        total_loss = L_w2v + 0.5 * L_distill + 0.1 * L_orth + 0.01 * L_ent + 0.001 * L2_reg

        return total_loss