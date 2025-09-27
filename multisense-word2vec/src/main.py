from sensate.pipeline import Trainer
from sensate.schema import TrainerConfig
from utils import load_data, load_config

if __name__ == "__main__":
    print("🚀 Starting MultiSense Word2Vec training...")
    
    # Load configuration
    config = load_config("config.yaml")
    print(f"📋 Config loaded: {config}")
    
    # Initialize trainer
    trainer = Trainer(config=config)
    
    # Load dataset with HuggingFace authentication
    print("📊 Loading dataset...")
    data = load_data("viethq1906", "skyserver-sql-dataset")
    
    # Start training
    print("🔥 Starting training...")
    trainer.fit(data)
    
    # Optional: Save model
    # trainer.save_model("output/model")
    
    print("✅ Training completed!")