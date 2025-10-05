from typing import List
import torch
from transformers import BertTokenizer, BertModel

from sensate.factory.common import device


class BERTExtractor:
    """
    BERT-based feature extractor that computes embeddings by averaging
    the last 4 hidden layers of BERT for a [MASK] token.
    
    Formula: h_t = 1/4 * Σ(ĥ_t) where ĥ_t are the last 4 layers
    """
    
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the BERT extractor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained BERT model from Hugging Face
        """
        print(f"🔧 Loading BERT model: {model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(
            model_name,
            output_hidden_states=True  # Required to get all layer outputs
        )
        
        # Move model to GPU if available
        self.model.to(device)
        self.model.eval()  # Set to evaluation mode
        
        print(f"✅ BERT model loaded successfully on {device}")
    
    def __call__(self, tokens: List[str]) -> list:
        """
        Extract BERT embeddings for the [MASK] token by averaging last 4 layers.
        
        Args:
            tokens: List of tokens (must contain exactly one [MASK] token)
            
        Returns:
            A 1D list of 768 float values representing the embedding
            
        Raises:
            ValueError: If no [MASK] token is found in the input
        """
        # Validate input
        if "[MASK]" not in tokens:
            raise ValueError("Input must contain a [MASK] token")
        
        # Find the position of [MASK] token in the original tokens
        mask_position = tokens.index("[MASK]")
        
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
        
        # Find the actual position of [MASK] in the tokenized sequence
        # This might differ from the original position due to subword tokenization
        input_ids = inputs["input_ids"][0]
        mask_token_id = self.tokenizer.mask_token_id
        tokenized_mask_position = (input_ids == mask_token_id).nonzero(as_tuple=True)[0][0].item()
        
        # Get the last 4 layers (layers -4, -3, -2, -1)
        # Note: hidden_states[0] is the embedding layer, so we skip it
        last_four_layers = hidden_states[-4:]
        
        # Extract embeddings for the [MASK] token from each of the last 4 layers
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



# Usage
# if __name__ == "__main__":
#     extractor = BERTExtractor()
#     embeddings = extractor(["SELECT", "[MASK]", "FROM", "<TAB>", "WHERE", "<COL>", "=", "<STR>", ";"])
#     print(embeddings) # Must return a list with 768 elements
#     print(len(embeddings)) # Should be 768