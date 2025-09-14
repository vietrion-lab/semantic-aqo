from sensate.pipeline import Trainer
from sensate.schema import TrainerConfig
from utils import load_data, load_config

if __name__ == "__main__":
    config = load_config("config.yaml")
    trainer = Trainer(config=config)
    
    # Example usage
    data = load_data("owner", "dataset_name")
    trainer.fit(data)
    trainer.save_model("model_path")