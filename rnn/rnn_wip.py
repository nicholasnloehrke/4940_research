import torch
import torch.nn as nn
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from dataclasses import dataclass
 from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SentimentRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc1 = nn.Linear()
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.fc(out)


class SentimentDataset(Dataset):
    def __init__(self, embeddings: list[torch.Tensor], labels: torch.Tensor):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embeddings[idx], self.labels[idx]


def prepare_data(df: pd.DataFrame, indices: list[int], config: Config) -> DataLoader:
    fold_df = df.iloc[indices].copy()
    fold_df["embedding"] = fold_df["embedding"].apply(lambda x: torch.tensor(x))
    features = list(fold_df["embedding"])
    labels = torch.tensor(fold_df["sentiment"].values)

    dataset = SentimentDataset(features, labels)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


def monitor_gradient_flow(model: nn.Module, writer: SummaryWriter, step: int) -> None:
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f"Gradients/{name}", param.grad, step)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> plt.Figure:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = np.nan_to_num(cm)

    figure = plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".2f", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion Matrix (Normalized)")
    plt.tight_layout()
    return figure


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float) -> plt.Figure:
    figure = plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    return figure


def plot_precision_recall_curve(precision: np.ndarray, recall: np.ndarray) -> plt.Figure:
    figure = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.tight_layout()
    return figure


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    writer: SummaryWriter = None,
    epoch: int = None,
) -> float:
    model.train()
    loss_accum = 0

    for batch_idx, (embeddings, labels) in enumerate(dataloader):
        embeddings, labels = embeddings.to(device), labels.to(device)
        outputs = model(embeddings.float())
        loss = criterion(outputs, labels)
        loss_accum += loss.item()

        optimizer.zero_grad()
        loss.backward()

        if writer is not None and epoch is not None:
            monitor_gradient_flow(model, writer, epoch * len(dataloader) + batch_idx)

        optimizer.step()

        if writer is not None and epoch is not None:
            writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + batch_idx)

    return loss_accum / len(dataloader)


def train_model(model: nn.Module, dataloader: DataLoader, config: Config, writer: SummaryWriter) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    print("Training model...")
    for epoch in range(config.epochs):
        average_loss = train_epoch(model, dataloader, optimizer, criterion, writer, epoch)
        print(f"Epoch [{epoch+1}/{config.epochs}], Loss: {average_loss:.4f}")


def evaluate_model(model: nn.Module, dataloader: DataLoader, writer: SummaryWriter = None, fold: int = None) -> float:
    model.eval()
    correct, total = 0, 0
    all_predictions, all_labels, all_probabilities = [], [], []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings.float())
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            predicted = torch.argmax(outputs, dim=1)
            all_probabilities.extend(probabilities)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate accuracy
    accuracy = correct / total
    writer.add_scalar("Accuracy/validation", accuracy, fold)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    cm_fig = plot_confusion_matrix(cm, class_names=["Negative", "Positive"])
    writer.add_figure(f"Confusion_Matrix_{fold}", cm_fig, global_step=fold)

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilities)
    roc_auc = auc(fpr, tpr)
    roc_fig = plot_roc_curve(fpr, tpr, roc_auc)
    writer.add_figure(f"ROC_Curve_{fold}", roc_fig, global_step=fold)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_labels, all_probabilities)
    pr_fig = plot_precision_recall_curve(precision, recall)
    writer.add_figure(f"Precision_Recall_Curve_{fold}", pr_fig, global_step=fold)

    return accuracy


def cross_validate(df: pd.DataFrame, config: Config, log_path: str = None) -> None:
    if log_path is not None:
        writer = SummaryWriter(log_path)
    
    kf = KFold(n_splits=config.folds, shuffle=True)
    accuracies = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(df)):
        model = SentimentRNN(config.input_size, config.hidden_size, config.layers, config.bidirectional).to(device)
        train_loader = prepare_data(df, train_indices, config)
        val_loader = prepare_data(df, val_indices, config)

        print(f"Training Fold {fold + 1}")
        train_model(model, train_loader, config, writer)

        accuracy = evaluate_model(model, val_loader, writer, fold)
        accuracies.append(accuracy)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

    print(f"Average Accuracy: {sum(accuracies) / len(accuracies):.4f}")
    if writer:
        writer.close()

def main(args):
    config = Config(
        hidden_size=args.hidden_size or Config.hidden_size,
        layers=args.layers or Config.layers,
        batch_size=args.batch_size or Config.batch_size,
        bidirectional=args.bidirectional,
        epochs=args.epochs or Config.epochs,
        learning_rate=args.learning_rate or Config.learning_rate,
        folds=args.folds or Config.folds,
    )

    print(f"Configuration: {config}")

    df = pd.read_pickle(args.embeddings_file)
    cross_validate(df, config, args.log_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment RNN Pipeline")

    parser.add_argument("embeddings_file", type=str, help="Input pickle file containing embeddings and sentiments")
    parser.add_argument("--hidden_size", type=int, help="Size of hidden state")
    parser.add_argument("--layers", type=int, help="Number of layers (stacked RNN cells)")
    parser.add_argument("--batch_size", type=int, help="Batch size for training")
    parser.add_argument("--bidirectional", action="store_true", help="Enable/disable bidirectional LSTM")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--folds", type=int, help="Number of folds used for training")
    parser.add_argument("--log_path", type=str, help="Log file path")

    args = parser.parse_args()

    main(args)
