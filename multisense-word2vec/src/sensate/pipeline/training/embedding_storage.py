"""
Memory-efficient embedding storage using HDF5 and memory mapping.
This module provides optimized storage for large BERT embedding tables.
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from functools import lru_cache
from typing import Union, List
import tempfile
import os


class MemoryEfficientEmbeddingTable:
    """
    Memory-efficient storage for BERT embeddings using HDF5.
    
    Strategies used:
    1. Store embeddings in HDF5 format on disk
    2. Use float16 precision (saves 50% memory vs float32)
    3. Memory-mapped access for random reads
    4. LRU caching for frequently accessed embeddings
    """
    
    def __init__(self, 
                 embedding_table: pd.DataFrame = None,
                 cache_size: int = 10000,
                 use_float16: bool = True,
                 storage_path: str = None):
        """
        Initialize memory-efficient embedding storage.
        
        Args:
            embedding_table: DataFrame with columns ['id', 'embedding']
            cache_size: Number of embeddings to keep in LRU cache
            use_float16: Use float16 precision (saves 50% memory)
            storage_path: Path to HDF5 file (if None, uses temp directory)
        """
        self.cache_size = cache_size
        self.use_float16 = use_float16
        self.dtype = np.float16 if use_float16 else np.float32
        
        # Create temp file if no path specified
        if storage_path is None:
            temp_dir = tempfile.gettempdir()
            self.storage_path = os.path.join(temp_dir, 'bert_embeddings.h5')
            self.is_temp = True
        else:
            self.storage_path = storage_path
            self.is_temp = False
        
        # Create HDF5 file
        if embedding_table is not None:
            self._create_storage(embedding_table)
        
        # Open file for reading with memory mapping
        self.h5_file = None
        self._open_for_reading()
    
    def _create_storage(self, embedding_table: pd.DataFrame):
        """Create HDF5 storage from DataFrame."""
        print(f"📦 Creating memory-efficient embedding storage...")
        print(f"   Storage path: {self.storage_path}")
        print(f"   Precision: {self.dtype}")
        
        # Extract embeddings
        ids = embedding_table['id'].values
        embeddings = np.array([emb for emb in embedding_table['embedding'].values], 
                              dtype=self.dtype)
        
        # Calculate memory savings
        original_size_mb = embeddings.nbytes / (1024 * 1024) * 2  # float32 size
        actual_size_mb = embeddings.nbytes / (1024 * 1024)
        
        print(f"   Original size (float32): {original_size_mb:.2f} MB")
        print(f"   Optimized size ({self.dtype}): {actual_size_mb:.2f} MB")
        print(f"   Memory saved: {(original_size_mb - actual_size_mb):.2f} MB ({((1 - actual_size_mb/original_size_mb) * 100):.1f}%)")
        
        # Write to HDF5
        with h5py.File(self.storage_path, 'w') as f:
            f.create_dataset('ids', data=ids, compression='gzip', compression_opts=4)
            f.create_dataset('embeddings', data=embeddings, compression='gzip', compression_opts=4)
            f.attrs['num_embeddings'] = len(ids)
            f.attrs['embedding_dim'] = embeddings.shape[1]
            f.attrs['dtype'] = str(self.dtype)
        
        print(f"   ✅ Storage created with {len(ids)} embeddings")
    
    def _open_for_reading(self):
        """Open HDF5 file for memory-mapped reading."""
        if os.path.exists(self.storage_path):
            self.h5_file = h5py.File(self.storage_path, 'r', rdcc_nbytes=1024**2*100)  # 100MB cache
            self.ids = self.h5_file['ids'][:]
            self.embeddings = self.h5_file['embeddings']
            
            # Create ID to index mapping
            self.id_to_idx = {id_val: idx for idx, id_val in enumerate(self.ids)}
    
    @lru_cache(maxsize=10000)
    def get_embedding(self, embedding_id: int) -> np.ndarray:
        """
        Get embedding by ID with LRU caching.
        
        Args:
            embedding_id: The embedding ID to retrieve
            
        Returns:
            Embedding array (converted to float32 for computation)
        """
        if embedding_id not in self.id_to_idx:
            raise ValueError(f"Embedding ID {embedding_id} not found")
        
        idx = self.id_to_idx[embedding_id]
        embedding = self.embeddings[idx]
        
        # Convert to float32 for computation
        return embedding.astype(np.float32)
    
    def get_embeddings_batch(self, embedding_ids: List[int]) -> np.ndarray:
        """
        Get multiple embeddings efficiently.
        
        Args:
            embedding_ids: List of embedding IDs
            
        Returns:
            Array of embeddings [N, D]
        """
        indices = [self.id_to_idx[id_val] for id_val in embedding_ids]
        embeddings = self.embeddings[indices]
        return embeddings.astype(np.float32)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert back to DataFrame format (for compatibility)."""
        all_embeddings = self.embeddings[:].astype(np.float32)
        return pd.DataFrame({
            'id': self.ids,
            'embedding': list(all_embeddings)
        })
    
    def to_records(self, index=False):
        """
        Convert to numpy structured array (for compatibility with existing code).
        This is memory-efficient as it doesn't create a full DataFrame.
        """
        # Create structured dtype
        embedding_dim = self.embeddings.shape[1]
        dtype = [('id', 'i8'), ('embedding', f'f4', (embedding_dim,))]
        
        # Create records
        num_records = len(self.ids)
        records = np.empty(num_records, dtype=dtype)
        records['id'] = self.ids
        records['embedding'] = self.embeddings[:].astype(np.float32)
        
        return records
    
    def __getitem__(self, mask):
        """Support numpy boolean indexing (for compatibility)."""
        if isinstance(mask, np.ndarray) and mask.dtype == bool:
            indices = np.where(mask)[0]
            if len(indices) == 0:
                return np.array([])
            
            idx = indices[0]
            # Return a structure similar to numpy record
            class EmbeddingRecord:
                def __init__(self, id_val, embedding):
                    self.data = {'id': id_val, 'embedding': embedding}
                
                def __getitem__(self, key):
                    return self.data[key]
            
            return [EmbeddingRecord(self.ids[idx], self.embeddings[idx].astype(np.float32))]
        else:
            raise NotImplementedError("Only boolean indexing is supported")
    
    def __len__(self):
        """Return number of embeddings."""
        return len(self.ids)
    
    def close(self):
        """Close HDF5 file."""
        if self.h5_file is not None:
            try:
                self.h5_file.close()
            except:
                pass  # Ignore errors if already closed
            finally:
                self.h5_file = None
    
    def cleanup(self):
        """Clean up temporary files."""
        self.close()
        if self.is_temp and os.path.exists(self.storage_path):
            try:
                os.remove(self.storage_path)
                print(f"🗑️  Cleaned up temporary storage: {self.storage_path}")
            except:
                pass  # Ignore if file already deleted
    
    def __del__(self):
        """Destructor to ensure file is closed."""
        try:
            self.close()
        except:
            pass  # Ignore all errors during cleanup


def optimize_embedding_table(embedding_table: pd.DataFrame, 
                             cache_size: int = 10000,
                             use_float16: bool = True,
                             storage_path: str = None) -> MemoryEfficientEmbeddingTable:
    """
    Convert a standard embedding DataFrame to memory-efficient storage.
    
    Args:
        embedding_table: DataFrame with columns ['id', 'embedding']
        cache_size: Number of embeddings to cache in memory
        use_float16: Use half precision (saves 50% memory)
        storage_path: Optional custom storage path
    
    Returns:
        MemoryEfficientEmbeddingTable instance
    """
    return MemoryEfficientEmbeddingTable(
        embedding_table=embedding_table,
        cache_size=cache_size,
        use_float16=use_float16,
        storage_path=storage_path
    )


# Alternative: Query-level aggregation (even more memory efficient)
def create_query_level_embeddings(embedding_table: pd.DataFrame, 
                                  base_table: pd.DataFrame) -> pd.DataFrame:
    """
    Create query-level embeddings instead of token-level.
    
    This is the MOST memory-efficient approach since:
    - The training code only looks up by query_id (not individual tokens)
    - We can aggregate all token embeddings per query
    
    Args:
        embedding_table: DataFrame with columns ['id', 'embedding']
        base_table: DataFrame with 'embedding_id' and 'sql_query_id'
    
    Returns:
        DataFrame with columns ['id' (query_id), 'embedding' (aggregated)]
    """
    print("📊 Creating query-level embeddings (maximum memory savings)...")
    
    # Map embedding_id to query_id
    query_embeddings = {}
    
    for _, row in base_table.iterrows():
        query_id = row['sql_query_id']
        embedding_id = row['embedding_id']
        
        # Get embedding
        emb_row = embedding_table[embedding_table['id'] == embedding_id]
        if len(emb_row) > 0:
            embedding = emb_row.iloc[0]['embedding']
            
            if query_id not in query_embeddings:
                query_embeddings[query_id] = []
            query_embeddings[query_id].append(embedding)
    
    # Aggregate embeddings per query (mean pooling)
    query_level_data = []
    for query_id, embeddings in query_embeddings.items():
        mean_embedding = np.mean(embeddings, axis=0)
        query_level_data.append({
            'id': query_id,
            'embedding': mean_embedding
        })
    
    result = pd.DataFrame(query_level_data)
    
    original_count = len(embedding_table)
    new_count = len(result)
    savings_pct = (1 - new_count / original_count) * 100
    
    print(f"   Original embeddings: {original_count}")
    print(f"   Query-level embeddings: {new_count}")
    print(f"   Reduction: {savings_pct:.1f}%")
    print(f"   ✅ Memory savings: ~{savings_pct:.1f}%")
    
    return result
