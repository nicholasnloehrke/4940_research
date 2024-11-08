import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import argparse
from pathlib import Path

from models.ffnn import FFNN
from config.config import config
from data.dataloader import create_hidden_states_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model: nn.Module, dataloader: DataLoader):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.ffnn.learning_rate)
    print("epoch, loss")
    for epoch in range(config.ffnn.epochs):
        for embedding, sentiment in dataloader:
            embedding, sentiment = embedding.to(device), sentiment.to(device)
            outputs = model(embedding.float())
            loss = criterion(outputs, sentiment)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{epoch}, {loss.item():.4f}")


def train_folds(args):
    df = pd.read_pickle(args.hidden_states_file)
    kf = KFold(n_splits=config.ffnn.folds, shuffle=True)

    best_accuracy = 0
    best_model = None

    metrics = []
    for train_indices, val_indices in kf.split(df):
        model = FFNN().to(device)
        train_loader = create_hidden_states_dataloader(df, train_indices)
        val_loader = create_hidden_states_dataloader(df, val_indices)

        train(model, train_loader)

        metric = evaluate(model, val_loader)
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
        torch.save(best_model.state_dict(), path)


def evaluate(model: nn.Module, dataloader: DataLoader):
    model.eval()
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for embedding, label in dataloader:
            embedding, label = embedding.to(device), label.to(device)

            outputs = model(embedding.float())

            all_labels.extend(label.cpu().numpy())
            all_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    metric = {}
    all_labels = np.array(all_labels)
    all_outputs = np.array(all_outputs)

    TN, FP, FN, TP = confusion_matrix(all_labels, all_outputs).ravel()

    metric = {
        "unweighted_accuracy": (TP + TN) / (TP + TN + FP + FN),
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "mcc": ((TP * TN) - (FP * FN)) / float(np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))),
    }

    return metric


def main(args):
    print("Train FFNN")
    train_folds(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FFNN")
    parser.add_argument("hidden_states_file", type=str, help="Input hidden states file in pickle format")
    parser.add_argument("-o", "--model_file", type=str, default="ffnn_model.model", help="Output model file")

    args = parser.parse_args()

    main(args)
