import torch
import torch.nn as nn
import pandas as pd
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Hyperparameters
INPUT_SIZE = 300
HIDDEN_SIZE = 256
LAYERS = 2
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.005
FOLDS = 5

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class for embeddings and labels
class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

# RNN model with bidirectional LSTM
class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2, 1)  # Adjusted for bidirectional output

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]  # Output from the last time step
        out = self.fc(out)
        return out

# Create DataLoader from DataFrame and indices
def create_dataloader(df, indices):
    fold_df = df.iloc[indices]
    fold_df["embedding"] = fold_df["embedding"].apply(lambda x: torch.tensor(x))
    X = list(fold_df["embedding"])
    y = torch.tensor(fold_df["sentiment"].values)

    dataset = SentimentDataset(X, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Train the model and log train losses/validation accuracies
def train_and_evaluate(model, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_accuracies = [], []

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings.float())
            loss = criterion(outputs.squeeze(), labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_losses.append(train_loss / len(train_loader))
        val_accuracy = evaluate(model, val_loader)
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {train_loss:.4f}, Val Acc: {val_accuracy * 100:.2f}%")

    # Plot training losses and validation accuracies
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss per Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Validation Accuracy per Epoch')
    plt.legend()
    plt.show()

# Evaluate the model and calculate accuracy
def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings.float())
            predicted = torch.round(torch.sigmoid(outputs.squeeze()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

# Evaluate with a confusion matrix for deeper insights
def evaluate_with_cm(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            outputs = model(embeddings.float())
            preds = torch.round(torch.sigmoid(outputs.squeeze())).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.show()

# Main function to handle cross-validation and logging
def main(args):
    df = pd.read_pickle(args.embeddings_file)
    kf = KFold(n_splits=FOLDS, shuffle=True)
    accuracies = []

    for train_indices, val_indices in kf.split(df):
        model = SentimentRNN(INPUT_SIZE, HIDDEN_SIZE, LAYERS).to(device)
        train_loader = create_dataloader(df, train_indices)
        val_loader = create_dataloader(df, val_indices)

        print("Training on fold...")
        train_and_evaluate(model, train_loader, val_loader)

        print("Evaluating on validation set")
        accuracy = evaluate(model, val_loader)
        accuracies.append(accuracy)

        print(f"Fold accuracy: {accuracy * 100:.2f}%")
        evaluate_with_cm(model, val_loader)

    mean_accuracy = sum(accuracies) / len(accuracies)
    print(f"Mean accuracy across all folds: {mean_accuracy * 100:.2f}%")

    # Plot accuracies across folds
    plt.plot(accuracies, marker='o')
    plt.title('Accuracy Across Folds')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.show()

# Entry point for the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sentiment RNN")
    parser.add_argument("embeddings_file", type=str, help="Input pickle file containing embeddings and sentiments")
    args = parser.parse_args()
    main(args)
