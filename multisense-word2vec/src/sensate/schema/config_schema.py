from pydantic import BaseModel

class TrainerConfig(BaseModel):
    epochs: int
    learning_rate: float
    batch_size: int
    window_size: int
    # Much more ....

class BaseTableEntry(BaseModel):
    id: int
    center_word: str
    context_word: str
    sql_query: str
    embedding_id: int
