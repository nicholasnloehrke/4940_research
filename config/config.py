from dataclasses import dataclass
from pathlib import Path


@dataclass
class RNN:
    # hyperparameters
    input_size: int = 300
    hidden_size: int = 32
    num_layers: int = 1
    bias: bool = True
    dropout: float = 0
    bidirectional: bool = False

    # training parameters
    epochs: int = 10
    folds: int = 8
    learning_rate: float = 0.0005
    batch_size: int = 1


@dataclass
class FFNN:
    # hyperparameters
    input_size: int = RNN.hidden_size * 400
    
    # training parameters
    epochs: int = 10
    folds: int = 8
    learning_rate: float = 0.0005


@dataclass
class Paths:
    fasttext: str = str(Path("fasttext/crawl-300d-2M-subword.bin").resolve())


@dataclass
class Config:
    rnn: RNN
    ffnn: FFNN
    paths: Paths


config = Config(rnn=RNN(), paths=Paths(), ffnn=FFNN())
