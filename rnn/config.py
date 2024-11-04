from dataclasses import dataclass


@dataclass
class Config:
    input_size: int = 300
    hidden_size: int = 32
    layers: int = 1
    batch_size: int = 1
    bidirectional: bool = True
    epochs: int = 10
    learning_rate: float = 0.0005
    folds: int = 3