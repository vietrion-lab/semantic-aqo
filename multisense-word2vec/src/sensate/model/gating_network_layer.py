import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GatingNetworkLayer(nn.Module):
    """
    Gating Network Layer to compute gating probabilities for sense selection.
    """
    def __init__(self, sigma: float = 0.001, d: int = 300):
        super(GatingNetworkLayer, self).__init__()
        self.sigma = sigma
        self.d = d
        
    def forward(self,
        center_pos,                   # [B]
        query_token_ids,              # [B, T]
        center_sense_embeddings,      # [B, K, D]
        context_sense_embeddings      # [B, T-1, K, D] - excludes center position
    ):
        batch_size = center_pos.shape[0]
        
        v = context_sense_embeddings.mean(dim=-2)  # [B, T-1, D]
        a = torch.matmul(center_sense_embeddings, v.transpose(1, 2)) / math.sqrt(self.d)  # [B, K, T-1]
        a = a.transpose(1, 2)  # [B, T-1, K]
        
        # Create position indices for context (excluding center)
        _, T = query_token_ids.shape
        pos = torch.arange(T, device=query_token_ids.device).unsqueeze(0).expand(batch_size, -1)  # [B, T]
        
        # Exclude center_pos from positions
        context_positions = []
        for b in range(batch_size):
            context_pos = torch.cat([
                pos[b, :center_pos[b]],
                pos[b, center_pos[b]+1:]
            ])  # [T-1]
            context_positions.append(context_pos)
        context_positions = torch.stack(context_positions, dim=0)  # [B, T-1]
        
        # Compute positional weights for context positions only
        w = torch.exp(- (context_positions - center_pos.unsqueeze(1)).abs() / self.sigma)  # [B, T-1]
        alpha = torch.softmax(a + (w.clamp_min(1e-12).log()).unsqueeze(-1), dim=1)  # [B, T-1, K]
        
        s = F.cosine_similarity(center_sense_embeddings, torch.matmul(alpha.transpose(1, 2), v), dim=-1)  # [B, K]
        q = torch.softmax(s, dim=-1)  # [B, K]
        
        idx = q.argmax(dim=-1)  # [B]

        max_pooled_embedding = center_sense_embeddings.gather(
            1, idx.view(-1, 1, 1).expand(-1, 1, center_sense_embeddings.size(-1))
        ).squeeze(1)  # [B, D]
        
        return max_pooled_embedding, q  # [B, D], [B, K]