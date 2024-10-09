import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# load review data with embeddings
data = pd.read_csv('review_data_with_embeddings.csv')

class ReviewDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # convert the list of embeddings to a 2D tensor
        embedding_tensor = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return embedding_tensor, label_tensor

# convert labels to tensor
labels = data['rating'].values

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(data['embeddings'].tolist(), labels, test_size=0.2, random_state=42)

# create datasets and loaders
train_dataset = ReviewDataset(X_train, y_train)
test_dataset = ReviewDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# define LSTM model
class SentimentLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SentimentLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(h_n[-1])  # use the last hidden state
        return out

# set parameters
input_size = 300  # FastText embedding size
hidden_size = 128
output_size = 5  # sentiment classes

# Create model
model = SentimentLSTM(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')