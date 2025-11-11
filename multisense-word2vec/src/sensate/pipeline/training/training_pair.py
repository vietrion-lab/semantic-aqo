from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sensate.pipeline.training.embedding_storage import MemoryEfficientEmbeddingTable

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the same length within a batch.
    """
    center_pos = torch.stack([item['center_pos'] for item in batch])
    context_ids = torch.stack([item['context_ids'] for item in batch])
    
    # Pad query_token_ids to max length in batch
    query_token_ids = pad_sequence(
        [item['query_token_ids'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    
    bert_embeddings = torch.stack([item['bert_embeddings'] for item in batch])
    
    return {
        'center_pos': center_pos,
        'context_ids': context_ids,
        'query_token_ids': query_token_ids,
        'bert_embeddings': bert_embeddings,
    }

class TrainingPairDataset(Dataset):
    def __init__(self,
            base_table: pd.DataFrame,
            vocab_table: pd.DataFrame,
            bert_embedding_table,
            query_table: pd.DataFrame
        ):
        print("🚀 Precomputing dataset for fast training...")
        
        # Precompute all data as tensors to avoid lookups during training
        n_samples = len(base_table)
        
        # Extract base table data
        center_pos_list = base_table['center_pos'].values
        context_ids_list = base_table['context_word_id'].values
        query_ids_list = base_table['sql_query_id'].values
        
        # Create word->id mapping for O(1) lookup
        word_to_id = dict(zip(vocab_table['word'].values, vocab_table['id'].values))
        
        # Create query_id->embedding mapping
        if isinstance(bert_embedding_table, MemoryEfficientEmbeddingTable):
            embedding_df = bert_embedding_table.to_dataframe()
        else:
            embedding_df = bert_embedding_table
        
        embedding_dict = dict(zip(embedding_df['id'].values, embedding_df['embedding'].values))
        
        # Precompute query token IDs (parse once, reuse forever)
        query_token_ids_dict = {}
        for _, row in tqdm(query_table.iterrows(), total=len(query_table), desc="Precomputing query tokens", unit="query"):
            query_id = row['id']
            tokens = row['sql_query'].split(' ')
            token_ids = [word_to_id[token] for token in tokens if token in word_to_id]
            query_token_ids_dict[query_id] = np.array(token_ids, dtype=np.int64)
        
        # Store precomputed tensors
        self.center_pos = torch.from_numpy(center_pos_list).long()
        self.context_ids = torch.from_numpy(context_ids_list).long()
        self.query_ids = query_ids_list
        self.query_token_ids_dict = query_token_ids_dict
        self.embedding_dict = embedding_dict
        
        print(f"   ✓ Precomputed {n_samples} samples")

    def __len__(self):
        return len(self.center_pos)

    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        
        return {
            'center_pos': self.center_pos[idx],
            'context_ids': self.context_ids[idx],
            'query_token_ids': torch.from_numpy(self.query_token_ids_dict[query_id]).long(),
            'bert_embeddings': torch.tensor(self.embedding_dict[query_id], dtype=torch.float32),
        }