import torch
import torch.nn as nn
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score
import numpy as np
from pathlib import Path
import os
from config import Config




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=False):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        fc_dim = hidden_size * 2 if bidirectional else hidden_size
        self.fc1 = nn.Linear(fc_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out


class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def create_dataloader(df: pd.DataFrame, indices, config: Config) -> DataLoader:
    fold_df = df.iloc[indices]
    fold_df.loc[:, "embedding"] = fold_df["embedding"].apply(lambda x: torch.tensor(x))
    features = list(fold_df["embedding"])
    labels = torch.tensor(fold_df["sentiment"].values)

    dataset = SentimentDataset(features, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=True)


def train(model: nn.Module, dataloader: DataLoader, config: Config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    print("epoch, loss")
    for epoch in range(config.epochs):
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings.float())
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{epoch}, {loss.item():.4f}")


def evaluate(model: nn.Module, dataloader: DataLoader, config: Config):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            outputs = model(embeddings.float())

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    metric = {}
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    matrix = confusion_matrix(all_labels, all_outputs)

    TN = matrix[0][0]
    FP = matrix[0][1]
    FN = matrix[1][0]
    TP = matrix[1][1]

    metric["unweighted_accuracy"] = (TP + TN) / (TP + TN + FP + FN)
    metric["TN"] = TN
    metric["FP"] = FP
    metric["FN"] = FN
    metric["TP"] = TP

    MCC = ((TP * TN) - (FP * FN)) / float(np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))

    metric["mcc"] = MCC

    return metric


def main(args):
    config = Config(
        hidden_size=args.hidden_size if args.hidden_size is not None else Config.hidden_size,
        layers=args.layers if args.layers is not None else Config.layers,
        batch_size=args.batch_size if args.batch_size is not None else Config.batch_size,
        bidirectional=args.bidirectional,
        epochs=args.epochs if args.epochs is not None else Config.epochs,
        learning_rate=args.learning_rate if args.learning_rate is not None else Config.learning_rate,
        folds=args.folds if args.folds is not None else Config.folds,
    )

    print(config)

    df = pd.read_pickle(args.embeddings_file)
    kf = KFold(n_splits=config.folds, shuffle=True)

    best_accuracy = 0
    best_model = None

    metrics = []
    for train_indices, val_indices in kf.split(df):
        model = SentimentRNN(config.input_size, config.hidden_size, config.layers, config.bidirectional).to(device)
        train_loader = create_dataloader(df, train_indices, config)
        val_loader = create_dataloader(df, val_indices, config)

        train(model, train_loader, config)

        metric = evaluate(model, val_loader, config)
        metrics.append(metric)

        if metric["unweighted_accuracy"] > best_accuracy:
            best_accuracy = metric["unweighted_accuracy"]
            best_model = model

        print(f'Unweighted accuracy: {metric["unweighted_accuracy"]:.3f}')
        print(f'TN: {metric["TN"]}')
        print(f'FP: {metric["FP"]}')
        print(f'FN: {metric["FN"]}')
        print(f'TP: {metric["TP"]}')
        print(f'mcc: {metric["mcc"]:.3f}')

    if args.model_file is not None:
        path = Path(args.model_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        best_model_scripted = torch.jit.script(best_model.to(device="cpu"))
        best_model_scripted.save(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment RNN")

    parser.add_argument("embeddings_file", type=str, help="Input pickle file containing embeddings and sentiments")
    parser.add_argument("--hidden_size", type=int, help="Size of hidden state")
    parser.add_argument("--layers", type=int, help="Number of layers (stacked RNN cells)")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--bidirectional", action="store_true", help="Enable/disable bidirectional LSTM")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--folds", type=int, help="Number of folds used for training")
    parser.add_argument("--model_file", type=str, help="Output file path to save model")

    args = parser.parse_args()

    main(args)
