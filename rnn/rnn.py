import torch
import torch.nn as nn
import pandas as pd
import argparse
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from dataclasses import dataclass
import os

log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuner.log')

logging.basicConfig(
    filename=log_file_path,  # Log file name
    level=logging.INFO,  # Set the logging level
    format='%(message)s'  # Format of the log messages
)

@dataclass
class Config:
    input_size: int = 300
    hidden_size: int = 64
    layers: int = 1
    batch_size: int = 256
    bidirectional: bool = True
    epochs: int = 2
    learning_rate: float = 0.0005
    folds: int = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class SentimentDataset(Dataset):
    """Map-style dataset"""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def create_dataloader(df: pd.DataFrame, indices, config: Config):
    fold_df = df.iloc[indices]
    fold_df.loc[:, "embedding"] = fold_df["embedding"].apply(lambda x: torch.tensor(x))
    features = list(fold_df["embedding"])  # features
    labels = torch.tensor(fold_df["sentiment"].values)  # labels

    dataset = SentimentDataset(features, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

def train(model: nn.Module, dataloader: DataLoader, config: Config):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    logging.info("epoch,loss")
    for epoch in range(config.epochs):
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings.float())
            loss = criterion(outputs.squeeze(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logging.info(f"{epoch},{loss.item():.4f}")

def evaluate(model: nn.Module, dataloader: DataLoader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            outputs = model(embeddings.float())
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    
    return accuracy

def main(args):
    config = Config(
        hidden_size=args.hidden_size if args.hidden_size is not None else Config.hidden_size,
        layers=args.layers if args.layers is not None else Config.layers,
        batch_size=args.batch_size if args.batch_size is not None else Config.batch_size,
        bidirectional=args.bidirectional,
        epochs=args.epochs if args.epochs is not None else Config.epochs,
        learning_rate=args.learning_rate if args.learning_rate is not None else Config.learning_rate,
        folds=args.folds if args.folds is not None else Config.folds
    )
    
    logging.info(config)

    df = pd.read_pickle(args.embeddings_file)
    kf = KFold(n_splits=config.folds, shuffle=True)
    accuracies = []
    
    for train_indices, val_indices in kf.split(df):
        model = SentimentRNN(config.input_size, config.hidden_size, config.layers, config.bidirectional).to(device)
        train_loader = create_dataloader(df, train_indices, config)
        val_loader = create_dataloader(df, val_indices, config)
        
        logging.info("__train_start__")
        train(model, train_loader, config)
        logging.info("__train_stop__")
        
        logging.info("__evaluate_start__")
        accuracy = evaluate(model, val_loader)
        accuracies.append(accuracy)
        
        logging.info(f"{accuracy:.2f}")
        logging.info("__evaluate_stop__")

if __name__ == "__main__":
    logging.info("__program_start__")
    parser = argparse.ArgumentParser(description="Sentiment RNN")
    parser.add_argument("embeddings_file", type=str, help="Input pickle file containing embeddings and sentiments")
    parser.add_argument("--hidden_size", type=int, help="Size of hidden state")
    parser.add_argument("--layers", type=int, help="Number of layers (stacked RNN cells)")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--bidirectional", action="store_true", help="Enable/disable bidirectional LSTM")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--folds", type=int, help="Number of folds used for training")
        
    args = parser.parse_args()
    
    main(args)
    logging.info("__program_stop__")
