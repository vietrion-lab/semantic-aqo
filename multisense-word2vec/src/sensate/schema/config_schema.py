from pydantic import BaseModel

class TrainerConfig(BaseModel):
    epochs: int
    learning_rate: float
    batch_size: int
    window_size: int
    # Much more ....