from typing import List
import torch
from transformers import RobertaTokenizer, RobertaModel

from sensate.factory.common import device


class BERTExtractor:
    """
    BERT-based feature extractor that computes embeddings by averaging
    the last 4 hidden layers of BERT for a <mask> token.
    
    Formula: h_t = 1/4 * Σ(ĥ_t) where ĥ_t are the last 4 layers
    """
    
    def __init__(self, model_name: str = None):
        """
        Initialize the BERT extractor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained BERT model from Hugging Face
        """
        assert model_name is not None, "A foundation model name must be provided"
        print(f"🔧 Loading BERT model: {model_name}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(
            model_name,
            output_hidden_states=True  # Required to get all layer outputs
        )
        
        # Move model to GPU if available
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ BERT model loaded successfully on {device}")
    
    def __call__(self, tokens: List[str]) -> list:
        """
        Extract BERT embeddings for the <mask> token by averaging last 4 layers.
        
        Args:
            tokens: List of tokens (must contain exactly one <mask> token)
            
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

        # Convert tokens to text
        text = " ".join(tokens)
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as model
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract hidden states from all layers
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.hidden_states
        
        # Find the actual position of <mask> in the tokenized sequence
        # This might differ from the original position due to subword tokenization
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
        
        # Convert to list and return
        return averaged_embedding.cpu().tolist()
    
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
        
        # Convert all token sequences to text
        texts = [" ".join(tokens) for tokens in batch_tokens]
        
        # Tokenize all inputs in batch
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as model
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        # Get model outputs for the entire batch
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Extract hidden states from all layers
        # hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape: (batch_size, sequence_length, hidden_size)
        hidden_states = outputs.hidden_states
        
        # Get the last 4 layers
        last_four_layers = hidden_states[-4:]
        
        # Process each item in the batch
        batch_embeddings = []
        mask_token_id = self.tokenizer.mask_token_id
        
        for batch_idx in range(len(batch_tokens)):
            # Find the position of <mask> in this sequence
            input_ids = inputs["input_ids"][batch_idx]
            mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]
            
            if len(mask_positions) == 0:
                raise ValueError(f"No <mask> token found in tokenized sequence {batch_idx}")
            
            tokenized_mask_position = mask_positions[0].item()
            
            # Extract embeddings for the <mask> token from each of the last 4 layers
            mask_embeddings = []
            for layer in last_four_layers:
                # layer shape: (batch_size, sequence_length, hidden_size)
                mask_embedding = layer[batch_idx, tokenized_mask_position, :]  # Shape: (768,)
                mask_embeddings.append(mask_embedding)
            
            # Stack and compute average: h_t = 1/4 * Σ(ĥ_t)
            stacked_embeddings = torch.stack(mask_embeddings)  # Shape: (4, 768)
            averaged_embedding = torch.mean(stacked_embeddings, dim=0)  # Shape: (768,)
            
            # Convert to list and add to batch results
            batch_embeddings.append(averaged_embedding.cpu().tolist())
        
        return batch_embeddings



# Usage
# if __name__ == "__main__":
#     extractor = BERTExtractor()
#     embeddings = extractor(["SELECT", "<mask>", "FROM", "<TAB>", "WHERE", "<COL>", "=", "<STR>", ";"])
#     print(embeddings) # Must return a list with 768 elements
#     print(len(embeddings)) # Should be 768