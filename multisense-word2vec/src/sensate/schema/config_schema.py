from pydantic import BaseModel

class TrainerConfig(BaseModel):
    epochs: int
    learning_rate: float
    batch_size: int
    window_size: int
    embedding_dim: int
    distill_dim: int
    num_senses: int
    # Much more ....
    
class FoundationConfigSchema(BaseModel):
    foundation_model: str
    # More foundation model related configs
    
class GlobalConfigSchema(BaseModel):
    training: TrainerConfig
    foundation: FoundationConfigSchema
    # model_config: 
    
class BaseTableEntry(BaseModel):
    id: int
    center_word_id: int
    context_word_id: int
    embedding_id: int
    sql_query_id: int
