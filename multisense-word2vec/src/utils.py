import yaml

from sensate.schema import TrainerConfig


def load_data(owner: str, dataset_name: str) -> list:
    # FAKE IMPLEMENTATION
    return [
        "SELECT * FROM users;", 
        "SELECT * FROM orders WHERE status = 'pending';", 
        "SELECT * FROM products ORDER BY price DESC;"
    ]
    
def load_config(path: str) -> TrainerConfig:
    with open(path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return TrainerConfig(**config_dict['training'])