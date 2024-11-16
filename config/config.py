from dataclasses import dataclass
from pathlib import Path


@dataclass
class RNN:
    # hyperparameters
    input_size: int = 300
    hidden_size: int = 50
    num_layers: int = 5
    bias: bool = True
    dropout: float = 0.1
    bidirectional: bool = False

    # training parameters
    epochs: int = 30
    folds: int = 2
    learning_rate: float = 0.1
    batch_size: int = 1


@dataclass
class FFNN:
    # hyperparameters
    hidden_state_embedding_size: int = 512
    input_size: int = RNN.hidden_size * hidden_state_embedding_size

    # training parameters
    epochs: int = 20
    folds: int = 2
    learning_rate: float = 0.001
    batch_size: int = 256


@dataclass
class Paths:
    fasttext: str = str(Path("fasttext/crawl-300d-2M-subword.bin").resolve())


@dataclass
class Config:
    rnn: RNN
    ffnn: FFNN
    paths: Paths


config = Config(rnn=RNN(), paths=Paths(), ffnn=FFNN())
