import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# Parameters
embedding_dim = 300  # Example: FastText embeddings
hidden_dim = 128     # LSTM hidden state size
batch_size = 32
sequence_length = 200  # Truncate/pad each review to this length
num_classes = 2       # Positive or negative sentiment
learning_rate = 1e-3
num_epochs = 5

# Step 1: Load and preprocess the CSV data
file_path = "reviews/raw_reviews.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Filter reviews based on ratings
df = df[(df['rating'] <= 2) | (df['rating'] >= 4)]
df['label'] = (df['rating'] >= 4).astype(int)  # 1 for positive, 0 for negative

# Split into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['review'].values, df['label'].values, test_size=0.2, random_state=42
)

# Tokenize and embed reviews
def tokenize_and_embed(reviews, vocab, embedding_matrix):
    tokenized_reviews = []
    for review in reviews:
        tokens = review.lower().split()[:sequence_length]  # Tokenize and truncate
        indices = [vocab.get(token, 0) for token in tokens]  # Map to vocab indices
        embeddings = embedding_matrix[indices]  # Map to embeddings
        if len(embeddings) < sequence_length:
            pad_size = sequence_length - len(embeddings)
            embeddings = np.vstack([embeddings, np.zeros((pad_size, embedding_dim))])
        tokenized_reviews.append(embeddings)
    return np.array(tokenized_reviews)

# Dummy vocabulary and embeddings (replace with FastText or similar)
vocab = {'example': 1, 'words': 2, 'go': 3, 'here': 4}  # Dummy vocab
embedding_matrix = np.random.randn(len(vocab) + 1, embedding_dim)  # Random embeddings

# Preprocess data
train_embeddings = tokenize_and_embed(train_texts, vocab, embedding_matrix)
val_embeddings = tokenize_and_embed(val_texts, vocab, embedding_matrix)

# Step 2: Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        final_hidden_state = hn[-1]  # Last hidden state
        output = self.fc(final_hidden_state)
        return output, lstm_out

# Step 3: Train the LSTM
class ReviewDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

train_dataset = ReviewDataset(train_embeddings, train_labels)
val_dataset = ReviewDataset(val_embeddings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(embedding_dim, hidden_dim, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for reviews, labels in train_loader:
        reviews, labels = reviews.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(reviews)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# Step 4: Extract Hidden States
def extract_hidden_states(loader, model):
    model.eval()
    all_hidden_states = []
    all_labels = []
    with torch.no_grad():
        for reviews, labels in loader:
            reviews = reviews.to(device)
            _, lstm_out = model(reviews)
            final_hidden_state = lstm_out[:, -1, :]  # Extract last hidden state
            all_hidden_states.append(final_hidden_state.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.vstack(all_hidden_states), np.hstack(all_labels)

train_hidden_states, train_hidden_labels = extract_hidden_states(train_loader, model)
val_hidden_states, val_hidden_labels = extract_hidden_states(val_loader, model)

# Step 5: Define the FFNN
class FFNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

ffnn = FFNN(hidden_dim, num_classes).to(device)
optimizer = torch.optim.Adam(ffnn.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

ffnn_dataset = ReviewDataset(train_hidden_states, train_hidden_labels)
ffnn_loader = DataLoader(ffnn_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    ffnn.train()
    running_loss = 0.0
    for features, labels in ffnn_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = ffnn(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"FFNN Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(ffnn_loader):.4f}")
