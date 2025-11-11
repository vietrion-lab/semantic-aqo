import torch
from accelerate import Accelerator
from tqdm import tqdm
import pandas as pd
import os

from sensate.pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from sensate.pipeline.preprocessing.training_sample_pipeline import TrainingSampleGenerator
from sensate.model.sensate import Sensate
from sensate.pipeline.training.training_pair import TrainingPairDataset, collate_fn

class Trainer:
    def __init__(self, config=None):
        assert config is not None, "Config must be provided, the format is defined in sensate.schema"
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.training_sample_generator = TrainingSampleGenerator(
            window_size=config.training.window_size, 
            foundation_model_name=config.foundation.foundation_model
        )
        self.base_table = None
        self.vocab_table = None
        self.embedding_table = None
        self.query_table = None
        self.model = None

    def prepare(self, data):
        preprocessed_data = self.preprocessing_pipeline(data)
        self.vocab_table, self.query_table, self.embedding_table, self.base_table = self.training_sample_generator(corpus=preprocessed_data)
        print(f"Corpus length: {len(self.base_table)}, Vocab size: {len(self.vocab_table)}")

        print("=" * 20)
        print(f"Base Table: \n{self.base_table.head()}")
        print(f"Base Table shape: {self.base_table.shape}")
        print(f"Vocab Table: \n{self.vocab_table.head()}")
        print(f"Vocab Table shape: {self.vocab_table.shape}")
        print(f"Embedding Table: \n{self.embedding_table.head()}")
        print(f"Embedding Table shape: {self.embedding_table.shape}")
        print(f"Query Table: \n{self.query_table.head()}")
        print(f"Query Table shape: {self.query_table.shape}")
        print("=" * 20)

    def fit(self):
        acce = Accelerator()
        self.model = Sensate(
            base_table=self.base_table,
            vocab_table=self.vocab_table,
            embedding_table=self.embedding_table,
            query_table=self.query_table,
            num_senses=self.config.training.num_senses,
            embedding_dim=self.config.training.embedding_dim
        )
        # Use the model's transformed embedding_table (150-dim after IPCA)
        dataset = TrainingPairDataset(
            base_table=self.base_table,
            vocab_table=self.vocab_table,
            bert_embedding_table=self.model.embedding_table,  # Use model's transformed version!
            query_table=self.query_table,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            collate_fn=collate_fn
        )
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.config.training.learning_rate)

        self.model, optimizer, dataloader = acce.prepare(self.model, opt, dataloader)
        self.model.train()
        
        for epoch in tqdm(range(self.config.training.num_epochs), desc="Training Epochs", unit="epoch"):
            epoch_loss = 0.0
            batch_count = 0
            
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.training.num_epochs}", leave=False, unit="batch"):
                optimizer.zero_grad()
                loss = self.model(
                    center_pos=batch['center_pos'],
                    context_ids=batch['context_ids'],
                    query_token_ids=batch['query_token_ids'],
                    bert_embeddings=batch['bert_embeddings']
                )
                acce.backward(loss)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            tqdm.write(f"Epoch {epoch+1}/{self.config.training.num_epochs} - Average Loss: {avg_loss:.4f}")

    def save_model(self, path=None):
        assert path is not None, "Path must be provided to save the model"
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary table
        vocab_path = os.path.join(path, 'vocab.csv')
        self.vocab_table.to_csv(vocab_path, index=False)
        print(f"✓ Vocabulary saved to {vocab_path}")
        print(f"  Total words: {len(self.vocab_table)}")
        
        # Get sense embeddings from model [V, K, D]
        sense_embeddings = self.model.sense_embeddings.detach().cpu().numpy()
        n_vocab, n_sense, embedding_dim = sense_embeddings.shape
        
        # Create word_id to word mapping
        word_id_to_word = dict(zip(self.vocab_table['id'], self.vocab_table['word']))
        
        # Prepare data for CSV
        records = []
        for word_id in range(n_vocab):
            word = word_id_to_word.get(word_id, f"<UNK_{word_id}>")  # Handle missing words
            for sense_id in range(n_sense):
                embedding = sense_embeddings[word_id, sense_id, :]
                record = {
                    'word': word,
                    'sense_id': sense_id,
                    'embedding': embedding.tolist()  # Convert to list for CSV storage
                }
                records.append(record)
        
        # Create DataFrame and save to CSV
        sense_df = pd.DataFrame(records)
        csv_path = os.path.join(path, 'sense_embeddings.csv')
        sense_df.to_csv(csv_path, index=False)
        print(f"✓ Sense embeddings saved to {csv_path}")
        print(f"  Total records: {len(records)} ({n_vocab} words × {n_sense} senses)")
        
        # Also save the model state dict
        model_path = os.path.join(path, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Model state dict saved to {model_path}")
        
        return csv_path