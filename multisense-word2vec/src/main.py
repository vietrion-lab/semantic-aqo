from sensate.pipeline import Trainer
from utils import load_data, load_config

if __name__ == "__main__":
    print("🚀 Starting MultiSense Word2Vec training...")
    
    # Initialize trainer
    config = load_config("config.yaml")
    trainer = Trainer(config=config)
    
    # Load dataset with HuggingFace authentication
    print("📊 Loading dataset...")
    data = load_data("viethq1906", "skyserver-sql-dataset")
    
    # Start training
    print("🔥 Starting training...")
    trainer.prepare(data)
    trainer.fit()
    trainer.save_model("../output/model")
    
    print("✅ Training completed!")