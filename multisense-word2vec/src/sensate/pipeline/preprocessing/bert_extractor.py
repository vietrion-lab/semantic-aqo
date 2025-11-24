from typing import List
import torch
import numpy as np
from transformers import RobertaTokenizer, RobertaModel

from sensate.factory.common import device


class BERTExtractor:
    """
    BERT-based feature extractor that computes embeddings by averaging
    the last 4 hidden layers of BERT for a <mask> token.
    
    Formula: h_t = 1/4 * Σ(ĥ_t) where ĥ_t are the last 4 layers
    Optionally projects to lower dimension using IPCA for RAM efficiency.
    """
    
    def __init__(self, model_name: str = None, ipca=None):
        """
        Initialize the BERT extractor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained BERT model from Hugging Face
            ipca: Optional IncrementalPCA model for dimension reduction
        """
        assert model_name is not None, "A foundation model name must be provided"
        print(f"🔧 Loading BERT model: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(
            model_name,
            output_hidden_states=True  # Required to get all layer outputs
        )
        
        # Move model to GPU if available and optimize for inference
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        # Enable TF32 for faster computation on Ampere GPUs (RTX 30xx, A100, etc.)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Optimize CUDA kernels
            torch.backends.cudnn.benchmark = True
        
        self.ipca = ipca
        
        if self.ipca is not None:
            print(f"✅ BERT model loaded with IPCA projection (768 -> {self.ipca.n_components} dim)")
        else:
            print(f"✅ BERT model loaded successfully on {device} with GPU optimizations")
    
    def __call__(self, tokens: List[str]) -> list:
        """
        Extract BERT embeddings for the [MASK] token by averaging last 4 layers.
        
        Args:
            tokens: List of tokens (must contain exactly one [MASK] token)
            
        Returns:
            A 1D list of 768 float values representing the embedding
            Raises:
                ValueError: If no <mask> token is found in the input
        """
        # Validate input
        if "<mask>" not in tokens:
            raise ValueError("Input must contain a <mask> token")

        # Find the position of <mask> token in the original tokens
        mask_position = tokens.index("<mask>")
        
        # Tokenize each token separately and combine
        tokenized_parts = []
        for i, token in enumerate(tokens):
            if i == mask_position:
                # Use the mask token ID directly
                tokenized_parts.append([self.tokenizer.mask_token_id])
            else:
                # Tokenize normally, remove special tokens
                tok = self.tokenizer.encode(token, add_special_tokens=False)
                if tok:
                    tokenized_parts.append(tok)
        
        # Flatten and add CLS token at start
        sequence_ids = [self.tokenizer.cls_token_id]
        for part in tokenized_parts:
            sequence_ids.extend(part)
        
        # Check if we need to truncate
        max_length = 510
        if len(sequence_ids) - 1 > max_length:
            # Find where the mask token is in the sequence_ids
            mask_position_in_seq = 1  # Start after CLS
            for i, part in enumerate(tokenized_parts):
                if i == mask_position:
                    break
                mask_position_in_seq += len(part)
            
            # Keep tokens around the mask position
            if mask_position_in_seq < max_length // 2:
                sequence_ids = sequence_ids[:max_length + 1]
            else:
                tokens_before = max_length // 2
                tokens_after = max_length - tokens_before
                start_pos = max(1, mask_position_in_seq - tokens_before)
                end_pos = min(len(sequence_ids), mask_position_in_seq + tokens_after)
                sequence_ids = [self.tokenizer.cls_token_id] + sequence_ids[start_pos:end_pos]
        
        # Add SEP token at end
        sequence_ids.append(self.tokenizer.sep_token_id)
        
        # Convert to tensor
        inputs = {
            "input_ids": torch.tensor([sequence_ids]),
            "attention_mask": torch.tensor([[1] * len(sequence_ids)])
        }
        
        # Move inputs to the same device as model
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get model outputs with mixed precision
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = self.model(**inputs)
        
        # Extract hidden states from all layers
        hidden_states = outputs.hidden_states
        
        # Find the actual position of mask token in the tokenized sequence
        input_ids = inputs["input_ids"][0]
        mask_token_id = self.tokenizer.mask_token_id
        tokenized_mask_position = (input_ids == mask_token_id).nonzero(as_tuple=True)[0][0].item()
        
        # Get the last 4 layers (layers -4, -3, -2, -1)
        # Note: hidden_states[0] is the embedding layer, so we skip it
        last_four_layers = hidden_states[-4:]
        
        # Extract embeddings for the <mask> token from each of the last 4 layers
        mask_embeddings = []
        for layer in last_four_layers:
            # layer shape: (batch_size, sequence_length, hidden_size)
            mask_embedding = layer[0, tokenized_mask_position, :]  # Shape: (768,)
            mask_embeddings.append(mask_embedding)
        
        # Stack and compute average: h_t = 1/4 * Σ(ĥ_t)
        stacked_embeddings = torch.stack(mask_embeddings)  # Shape: (4, 768)
        averaged_embedding = torch.mean(stacked_embeddings, dim=0)  # Shape: (768,)
        
        # Convert to numpy and apply IPCA if available
        embedding_np = averaged_embedding.cpu().numpy()
        
        if self.ipca is not None:
            embedding_np = self.ipca.transform(embedding_np.reshape(1, -1))[0]
        
        # Convert to list and return
        return embedding_np.tolist()
    
    def batch_extract(self, batch_tokens: List[List[str]]) -> List[list]:
        """
        Extract BERT embeddings for a batch of token sequences.
        Uses GPU parallelization for efficient processing.
        
        Args:
            batch_tokens: List of token lists, each containing exactly one <mask> token
            
        Returns:
            List of embeddings, each a 1D list of 768 float values
        """
        # Validate inputs
        for tokens in batch_tokens:
            if "<mask>" not in tokens:
                raise ValueError("Each input must contain a <mask> token")
        
        # Process each sequence: tokenize each token individually, then combine
        all_input_ids = []
        all_attention_masks = []
        
        for tokens in batch_tokens:
            # Find mask position
            mask_idx = tokens.index("<mask>")
            
            # Tokenize each token separately
            tokenized_parts = []
            for i, token in enumerate(tokens):
                if i == mask_idx:
                    # Use the mask token ID directly
                    tokenized_parts.append([self.tokenizer.mask_token_id])
                else:
                    # Tokenize normally, remove special tokens (CLS, SEP)
                    tok = self.tokenizer.encode(token, add_special_tokens=False)
                    if tok:  # Only add if tokenization produced something
                        tokenized_parts.append(tok)
            
            # Flatten and add CLS token at start
            sequence_ids = [self.tokenizer.cls_token_id]
            for part in tokenized_parts:
                sequence_ids.extend(part)
            
            # Check if we need to truncate (max length is 512 including CLS and SEP)
            max_length = 510  # Leave room for CLS and SEP
            if len(sequence_ids) - 1 > max_length:  # -1 because we already added CLS
                # Find where the mask token is in the sequence_ids
                mask_position_in_seq = None
                current_pos = 1  # Start after CLS
                for i, part in enumerate(tokenized_parts):
                    if i == mask_idx:
                        mask_position_in_seq = current_pos
                        break
                    current_pos += len(part)
                
                # If mask is in first half, keep first max_length tokens
                # If mask is in second half, keep tokens around the mask
                if mask_position_in_seq < max_length // 2:
                    # Mask is early, keep beginning
                    sequence_ids = sequence_ids[:max_length + 1]  # +1 for CLS
                else:
                    # Mask is late, keep tokens around mask position
                    # Calculate how many tokens to keep before and after mask
                    tokens_before = max_length // 2
                    tokens_after = max_length - tokens_before
                    start_pos = max(1, mask_position_in_seq - tokens_before)
                    end_pos = min(len(sequence_ids), mask_position_in_seq + tokens_after)
                    
                    # Rebuild sequence with CLS, selected tokens, and ensure mask is included
                    sequence_ids = [self.tokenizer.cls_token_id] + sequence_ids[start_pos:end_pos]
            
            # Add SEP token at end
            sequence_ids.append(self.tokenizer.sep_token_id)
            
            all_input_ids.append(sequence_ids)
        
        # Pad sequences to the same length
        max_len = max(len(seq) for seq in all_input_ids)
        padded_input_ids = []
        attention_masks = []
        
        for seq in all_input_ids:
            padding_length = max_len - len(seq)
            padded_input_ids.append(seq + [self.tokenizer.pad_token_id] * padding_length)
            attention_masks.append([1] * len(seq) + [0] * padding_length)
        
        # Convert to tensors
        inputs = {
            "input_ids": torch.tensor(padded_input_ids),
            "attention_mask": torch.tensor(attention_masks)
        }
        
        # Move inputs to the same device as model
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get model outputs for the entire batch with mixed precision for speed
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            outputs = self.model(**inputs)
        
        # Extract hidden states from all layers
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.hidden_states
        
        # Get the last 4 layers and stack them efficiently
        last_four_layers = torch.stack(hidden_states[-4:])  # Shape: (4, batch_size, seq_len, hidden_size)
        
        # Process each item in the batch
        batch_embeddings = []
        mask_token_id = self.tokenizer.mask_token_id
        
        for batch_idx in range(len(batch_tokens)):
            # Find the position of mask token in this sequence
            input_ids = inputs["input_ids"][batch_idx]
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_positions) == 0:
                # Debug: print what we got
                print(f"\nDEBUG - Sequence {batch_idx}:")
                print(f"  Original tokens: {batch_tokens[batch_idx]}")
                print(f"  Input IDs: {input_ids.tolist()}")
                print(f"  Looking for mask_token_id: {mask_token_id}")
                print(f"  Mask token string: {self.tokenizer.mask_token}")
                print(f"  Decoded: {self.tokenizer.decode(input_ids)}")
                raise ValueError(f"No mask token (ID={mask_token_id}) found in tokenized sequence {batch_idx}")
            
            tokenized_mask_position = mask_positions[0].item()
            
            # Extract embeddings for the mask token from last 4 layers efficiently
            # Shape: (4, hidden_size) - get mask position across all 4 layers at once
            mask_embeddings = last_four_layers[:, batch_idx, tokenized_mask_position, :]
            
            # Compute average: h_t = 1/4 * Σ(ĥ_t)
            averaged_embedding = torch.mean(mask_embeddings, dim=0)  # Shape: (768,)
            
            # Convert to numpy and add to batch results
            batch_embeddings.append(averaged_embedding.cpu().float().numpy())
        
        # Apply IPCA projection to entire batch if available
        if self.ipca is not None:
            batch_embeddings = self.ipca.transform(np.array(batch_embeddings))
        
        # Convert to list
        return [emb.tolist() for emb in batch_embeddings]



# Usage
# if __name__ == "__main__":
#     extractor = BERTExtractor()
#     embeddings = extractor(["SELECT", "[MASK]", "FROM", "<TAB>", "WHERE", "<COL>", "=", "<STR>", ";"])
#     print(embeddings) # Must return a list with 768 elements
#     print(len(embeddings)) # Should be 768