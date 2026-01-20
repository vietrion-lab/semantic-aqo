import torch
from accelerate import Accelerator
from tqdm import tqdm
import pandas as pd
import os
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import f1_score

from sensate.pipeline.preprocessing.preprocessing_pipeline import PreprocessingPipeline
from sensate.pipeline.preprocessing.training_sample_pipeline import TrainingSampleGenerator
from sensate.model.sensate import Sensate
from sensate.pipeline.training.evaluating_pipeline import Evaluator
from sensate.pipeline.training.training_pair import TrainingPairDataset, collate_fn

class Trainer:
    def __init__(self, config=None):
        assert config is not None, "Config must be provided, the format is defined in sensate.schema"
        self.config = config
        self.preprocessing_pipeline = PreprocessingPipeline()
        self.evaluator = Evaluator()
        self.ipca = None
        self.training_sample_generator = None
        self.base_table = None
        self.vocab_table = None
        self.embedding_table = None
        self.query_table = None
        self.model = None
        
        # Tracking mechanism
        self.history = {
            'epoch': [],
            'train_loss': [],
            'bombay_f1': [],
            'googleplus_f1': [],
            'ub_f1': [],
            'avg_f1': []
        }
        self.best_avg_f1 = -1.0
        self.best_epoch = -1
        self.checkpoint_dir = None

    def _fit_ipca_on_sample(self, preprocessed_data):
        """Fit IPCA on a small sample to reduce memory usage."""
        print(f"🔧 Fitting IPCA on sample data...")
        
        # Create a temporary generator without IPCA to get sample embeddings
        temp_generator = TrainingSampleGenerator(
            window_size=self.config.training.window_size, 
            foundation_model_name=self.config.foundation.foundation_model,
            ipca=None
        )
        
        # Use sample for fitting IPCA
        sample_size = min(50, len(preprocessed_data))
        sample_data = preprocessed_data[:sample_size]
        
        _, _, sample_embedding_table, _ = temp_generator(corpus=sample_data)
        
        # Extract embeddings and fit IPCA incrementally to avoid OOM
        print(f"   Sample queries: {sample_size}")
        print(f"   Fitting IPCA: 768 -> {self.config.training.embedding_dim} dimensions")
        
        ipca = IncrementalPCA(n_components=self.config.training.embedding_dim, batch_size=256)
        
        # Fit incrementally in batches
        batch_size = 512
        total_embeddings = len(sample_embedding_table)
        print(f"   Total sample embeddings: {total_embeddings}")
        
        for i in range(0, total_embeddings, batch_size):
            batch_df = sample_embedding_table.iloc[i:i+batch_size]
            batch_embeddings = np.array(batch_df['embedding'].tolist())
            ipca.partial_fit(batch_embeddings)
            print(f"   Fitted batch {i//batch_size + 1}/{(total_embeddings + batch_size - 1)//batch_size}")
        print(f"   ✓ IPCA fitted on sample")
        
        return ipca

    def prepare(self, data):
        preprocessed_data = self.preprocessing_pipeline(data)
        
        # Fit IPCA on sample first
        self.ipca = self._fit_ipca_on_sample(preprocessed_data)
        
        # Create generator with IPCA for full dataset
        self.training_sample_generator = TrainingSampleGenerator(
            window_size=self.config.training.window_size, 
            foundation_model_name=self.config.foundation.foundation_model,
            ipca=self.ipca
        )
        
        print(f"🔄 Processing full dataset with dimension reduction...")
        self.vocab_table, self.query_table, self.embedding_table, self.base_table = self.training_sample_generator(corpus=preprocessed_data)
        
        # Print retained variance after processing full dataset
        retained_var = self.ipca.explained_variance_ratio_.sum()
        print(f"   ✓ Retained variance after full dataset: {retained_var:.4f}")
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
    
    def _extract_f1_from_report(self, report_str):
        """Extract weighted avg F1-score from classification report string."""
        try:
            lines = report_str.strip().split('\n')
            for line in lines:
                if 'weighted avg' in line or 'macro avg' in line:
                    parts = line.split()
                    # Find f1-score column (usually 4th number)
                    f1 = float(parts[-2])
                    return f1
            # Fallback: return 0 if can't parse
            return 0.0
        except:
            return 0.0
    
    def _evaluate_model(self):
        """Run evaluation on all datasets and return F1 scores."""
        print("🔍 Evaluating model...", end=' ', flush=True)
        
        # Create vocab mapping
        vocab_dict = dict(zip(self.vocab_table['word'], self.vocab_table['id']))
        
        # Get sense embeddings from model
        sense_embeddings = self.model.sense_embeddings.detach()
        
        # Suppress sklearn warnings about ill-defined metrics (common in small datasets)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            # Run evaluation
            result_dict = self.evaluator(sense_embeddings, vocab_dict)
        
        # Extract F1 scores
        bombay_f1 = self._extract_f1_from_report(result_dict['bombay_report'])
        googleplus_f1 = self._extract_f1_from_report(result_dict['googleplus_report'])
        ub_f1 = self._extract_f1_from_report(result_dict['ub_report'])
        avg_f1 = (bombay_f1 + googleplus_f1 + ub_f1) / 3.0
        
        print(f"Done! Avg F1: {avg_f1:.4f}")
        
        return {
            'bombay_f1': bombay_f1,
            'googleplus_f1': googleplus_f1,
            'ub_f1': ub_f1,
            'avg_f1': avg_f1,
            'reports': result_dict
        }
    
    def _save_checkpoint(self, epoch, avg_f1, is_best=False):
        """Save model checkpoint."""
        checkpoint_name = f"checkpoint_epoch_{epoch}_f1_{avg_f1:.4f}.pt"
        if is_best:
            checkpoint_name = f"BEST_checkpoint_epoch_{epoch}_f1_{avg_f1:.4f}.pt"
        
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'avg_f1': avg_f1,
            'history': self.history,
            # Don't save pandas DataFrames - save as dict or CSV separately
            'vocab_size': len(self.vocab_table),
            'num_senses': self.config.training.num_senses,
            'embedding_dim': self.config.training.embedding_dim
        }
        
        torch.save(checkpoint, checkpoint_path)
        if is_best:
            print(f"🏆 New best checkpoint saved: F1={avg_f1:.4f}")
        
        # Save vocab_table separately as CSV (only for best checkpoint to save space)
        if is_best:
            vocab_csv_path = os.path.join(self.checkpoint_dir, f"vocab_best_epoch_{epoch}.csv")
            self.vocab_table.to_csv(vocab_csv_path, index=False)
        
        return checkpoint_path
    
    def _plot_training_progress(self, save_path):
        """Plot training loss and evaluation F1 scores."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs = self.history['epoch']
        
        # Plot 1: Training Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', linewidth=2, marker='o')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average F1 Score
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['avg_f1'], 'g-', linewidth=2, marker='o', label='Avg F1')
        if self.best_epoch >= 0:
            ax2.axvline(x=self.best_epoch, color='r', linestyle='--', label=f'Best (Epoch {self.best_epoch})')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Average F1 Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Individual Dataset F1 Scores
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['bombay_f1'], 'r-', linewidth=2, marker='s', label='Bombay')
        ax3.plot(epochs, self.history['googleplus_f1'], 'b-', linewidth=2, marker='^', label='GooglePlus')
        ax3.plot(epochs, self.history['ub_f1'], 'g-', linewidth=2, marker='D', label='UB')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.set_title('F1 Scores by Dataset')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Summary Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Find best epoch index (0-indexed) for accessing history lists
        best_idx = self.best_epoch - 1 if self.best_epoch > 0 else -1
        
        summary_text = f"""
        Training Summary
        ════════════════════════════
        Total Epochs: {len(epochs)}
        
        Best Performance:
        ├─ Epoch: {self.best_epoch}
        ├─ Avg F1: {self.best_avg_f1:.4f}
        └─ Final Loss: {self.history['train_loss'][best_idx]:.4f}
        
        Final Performance:
        ├─ Bombay F1: {self.history['bombay_f1'][-1]:.4f}
        ├─ GooglePlus F1: {self.history['googleplus_f1'][-1]:.4f}
        ├─ UB F1: {self.history['ub_f1'][-1]:.4f}
        └─ Avg F1: {self.history['avg_f1'][-1]:.4f}
        """
        ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace', 
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training progress plot saved to {save_path}")
        plt.close()
    
    def _save_history(self, save_path):
        """Save training history to JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📝 Training history saved to {save_path}")

    def fit(self, checkpoint_dir='checkpoints', eval_every_n_epochs=1):
        """
        Train the model with evaluation tracking and checkpointing.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            eval_every_n_epochs: Evaluate every N epochs (default: 1)
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        acce = Accelerator(
            mixed_precision='fp16',
            cpu=False,
            split_batches=False
        )
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
            bert_embedding_table=self.model.embedding_table,
            query_table=self.query_table,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.training.batch_size * 2,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
            collate_fn=collate_fn
        )
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            fused=torch.cuda.is_available()
        )
        
        # Learning rate scheduler - Reduce on plateau for adaptive learning
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=0.5, patience=2, min_lr=1e-6
        )

        self.model, optimizer, dataloader = acce.prepare(self.model, opt, dataloader)
        
        print("\n" + "="*60)
        print("🚀 Starting Training with Evaluation Tracking")
        print("="*60)
        
        for epoch in range(self.config.training.num_epochs):
            epoch_loss = 0.0
            batch_count = 0
            
            # Training phase
            self.model.train()
            pbar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{self.config.training.num_epochs}]", 
                       unit="batch", mininterval=1.0, ncols=150)
            
            for batch in pbar:
                optimizer.zero_grad(set_to_none=True)
                loss = self.model(
                    center_pos=batch['center_pos'],
                    context_ids=batch['context_ids'],
                    query_token_ids=batch['query_token_ids'],
                    bert_embeddings=batch['bert_embeddings']
                )
                acce.backward(loss)
                # Gradient clipping for stability
                acce.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
                # Update progress bar with all loss components
                avg_loss = epoch_loss / batch_count
                lr = optimizer.param_groups[0]['lr']
                
                if hasattr(self.model, 'last_loss_components'):
                    comp = self.model.last_loss_components
                    pbar.set_postfix_str(
                        f"Loss={avg_loss:.4f} | LR={lr:.6f} | W2V={comp['L_w2v']:.3f} | "
                        f"Dist={comp['L_distill']:.3f} | Orth={comp['L_orth']:.4f}"
                    )
                else:
                    pbar.set_postfix_str(f"Loss={avg_loss:.4f} | LR={lr:.6f}")
            
            pbar.close()
            
            avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            
            # Evaluation phase (every N epochs)
            should_eval = (epoch + 1) % eval_every_n_epochs == 0 or (epoch + 1) == self.config.training.num_epochs
            
            if should_eval:
                self.model.eval()
                with torch.no_grad():
                    eval_results = self._evaluate_model()
                
                # Update history
                self.history['epoch'].append(epoch + 1)
                self.history['train_loss'].append(avg_loss)
                self.history['bombay_f1'].append(eval_results['bombay_f1'])
                self.history['googleplus_f1'].append(eval_results['googleplus_f1'])
                self.history['ub_f1'].append(eval_results['ub_f1'])
                self.history['avg_f1'].append(eval_results['avg_f1'])
                
                # Check if this is the best model
                is_best = eval_results['avg_f1'] > self.best_avg_f1
                if is_best:
                    self.best_avg_f1 = eval_results['avg_f1']
                    self.best_epoch = epoch + 1
                
                # Print results
                print("\n" + "─"*60)
                print(f"📊 Epoch {epoch+1}/{self.config.training.num_epochs} Results:")
                print(f"   Loss: {avg_loss:.4f}")
                print(f"   Bombay F1:     {eval_results['bombay_f1']:.4f}")
                print(f"   GooglePlus F1: {eval_results['googleplus_f1']:.4f}")
                print(f"   UB F1:         {eval_results['ub_f1']:.4f}")
                print(f"   ➤ Avg F1:      {eval_results['avg_f1']:.4f} {'🏆 NEW BEST!' if is_best else ''}")
                print("─"*60 + "\n")
                
                # Save checkpoint
                self._save_checkpoint(epoch + 1, eval_results['avg_f1'], is_best=is_best)
                
                # Step the scheduler based on avg_f1 (ReduceLROnPlateau needs metric)
                scheduler.step(eval_results['avg_f1'])
            else:
                # Just record training loss
                self.history['epoch'].append(epoch + 1)
                self.history['train_loss'].append(avg_loss)
                print(f"Epoch {epoch+1}/{self.config.training.num_epochs} - Loss: {avg_loss:.4f}")
            
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
        
        # Save final visualization and history
        print("\n" + "="*60)
        print("✅ Training Completed!")
        print(f"🏆 Best Avg F1: {self.best_avg_f1:.4f} at Epoch {self.best_epoch}")
        print("="*60 + "\n")
        
        plot_path = os.path.join(checkpoint_dir, 'training_progress.png')
        self._plot_training_progress(plot_path)
        
        history_path = os.path.join(checkpoint_dir, 'training_history.json')
        self._save_history(history_path)

    def save_model(self, path=None, use_best_checkpoint=True):
        """
        Export model embeddings to human-readable format (CSV).
        
        Args:
            path: Directory to save the model files
            use_best_checkpoint: If True, loads the best checkpoint before exporting
        """
        assert path is not None, "Path must be provided to save the model"
        
        # Load best checkpoint if requested and available
        if use_best_checkpoint and self.best_epoch > 0 and self.checkpoint_dir:
            print(f"\n📦 Loading best checkpoint (Epoch {self.best_epoch}, F1: {self.best_avg_f1:.4f})...")
            # Find best checkpoint file
            best_checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.startswith('BEST_')]
            if best_checkpoint_files:
                best_checkpoint_path = os.path.join(self.checkpoint_dir, best_checkpoint_files[0])
                checkpoint = torch.load(best_checkpoint_path, weights_only=False)  # Allow non-tensor objects
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✓ Loaded best checkpoint from epoch {checkpoint['epoch']}")
            else:
                print("⚠️  No best checkpoint found, using current model state")
        
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save vocabulary table
        vocab_path = os.path.join(path, 'vocab.csv')
        self.vocab_table.to_csv(vocab_path, index=False)
        print(f"✓ Vocabulary saved to {vocab_path}")
        print(f"  Total words: {len(self.vocab_table)}")
        
        # Save vocabulary as binary format for PostgreSQL
        vocab_bin_path = os.path.join(path, 'vocab.bin')
        with open(vocab_bin_path, 'wb') as f:
            # Write number of records
            f.write(len(self.vocab_table).to_bytes(4, byteorder='little'))
            for _, row in self.vocab_table.iterrows():
                # Write word length and word (UTF-8 encoded)
                word_bytes = row['word'].encode('utf-8')
                f.write(len(word_bytes).to_bytes(4, byteorder='little'))
                f.write(word_bytes)
                # Write word id
                f.write(int(row['id']).to_bytes(4, byteorder='little'))
        print(f"✓ Vocabulary binary saved to {vocab_bin_path}")
        
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
        
        # Save sense embeddings as binary format for PostgreSQL
        bin_path = os.path.join(path, 'sense_embeddings.bin')
        with open(bin_path, 'wb') as f:
            # Write metadata: number of records, embedding dimension
            f.write(len(records).to_bytes(4, byteorder='little'))
            f.write(embedding_dim.to_bytes(4, byteorder='little'))
            
            # Write each record
            for record in records:
                # Write word length and word (UTF-8 encoded)
                word_bytes = record['word'].encode('utf-8')
                f.write(len(word_bytes).to_bytes(4, byteorder='little'))
                f.write(word_bytes)
                
                # Write sense_id
                f.write(int(record['sense_id']).to_bytes(4, byteorder='little'))
                
                # Write embedding as float32
                embedding_array = np.array(record['embedding'], dtype=np.float32)
                f.write(embedding_array.tobytes())
        
        print(f"✓ Sense embeddings binary saved to {bin_path}")
        print(f"  Binary file size: {os.path.getsize(bin_path) / (1024*1024):.2f} MB")
        
        # Also save the model state dict
        model_path = os.path.join(path, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        print(f"✓ Model state dict saved to {model_path}")
        
        # Copy training history and plot if available
        if self.checkpoint_dir and os.path.exists(self.checkpoint_dir):
            import shutil
            history_src = os.path.join(self.checkpoint_dir, 'training_history.json')
            plot_src = os.path.join(self.checkpoint_dir, 'training_progress.png')
            
            if os.path.exists(history_src):
                shutil.copy(history_src, os.path.join(path, 'training_history.json'))
                print(f"✓ Training history copied to {path}")
            
            if os.path.exists(plot_src):
                shutil.copy(plot_src, os.path.join(path, 'training_progress.png'))
                print(f"✓ Training plot copied to {path}")
        
        print(f"\n🎉 Model exported successfully to {path}")
        if use_best_checkpoint and self.best_epoch > 0:
            print(f"   Using best checkpoint from Epoch {self.best_epoch} (Avg F1: {self.best_avg_f1:.4f})")
        
        return csv_path