from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
import pandas as pd
import numpy as np

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the same length within a batch.
    """
    center_pos = torch.stack([item['center_pos'] for item in batch])
    
    # Stack context_ids - already single values per sample
    context_ids = torch.stack([item['context_ids'] for item in batch])
    
    # Pad query_token_ids to max length in batch
    query_token_ids = pad_sequence(
        [item['query_token_ids'] for item in batch],
        batch_first=True,
        padding_value=0
    )
    
    # Stack BERT embeddings
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
            bert_embedding_table: pd.DataFrame,
            query_table: pd.DataFrame
        ):
        # Convert to structured arrays for efficient access
        self.base_table = base_table.to_records(index=False)
        self.vocab_table = vocab_table.to_records(index=False)
        self.bert_embedding_table = bert_embedding_table.to_records(index=False)
        self.query_table = query_table.to_records(index=False)

    def __len__(self):
        return len(self.base_table)

    def __getitem__(self, idx):
        base_row = self.base_table[idx]
        center_id = base_row['center_word_id']
        center_pos = base_row['center_pos']
        context_id = base_row['context_word_id']  # Single context ID from base_table
        query_id = base_row['sql_query_id']
        
        # Get query information
        query_mask = self.query_table['id'] == query_id
        query_row = self.query_table[query_mask][0]
        tokens = query_row['sql_query'].split(' ')
        
        # Get token IDs for the query
        query_token_ids = []
        for token in tokens:
            token_mask = self.vocab_table['word'] == token
            if np.any(token_mask):
                query_token_ids.append(self.vocab_table[token_mask][0]['id'])
        query_token_ids = np.array(query_token_ids, dtype=np.int64)
        
        # Get BERT embedding for this query
        bert_mask = self.bert_embedding_table['id'] == query_id
        bert_embedding = self.bert_embedding_table[bert_mask][0]['embedding']

        row = {
            'center_pos': torch.tensor(center_pos, dtype=torch.long),
            'context_ids': torch.tensor(context_id, dtype=torch.long),  # Single context ID
            'query_token_ids': torch.tensor(query_token_ids, dtype=torch.long),
            'bert_embeddings': torch.tensor(bert_embedding, dtype=torch.float32),
        }

        return row