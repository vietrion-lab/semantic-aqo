from pydantic import BaseModel

class TrainerConfig(BaseModel):
    epochs: int
    learning_rate: float
    batch_size: int
    window_size: int
    # Much more ....

class BaseTableEntry(BaseModel):
    id: int
    center_word_id: int
    context_word_id: int
    embedding_id: int
    sql_query_id: int
