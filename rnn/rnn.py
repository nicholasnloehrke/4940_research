import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Hyperparameters
input_size = 300  # Embedding dimension
hidden_size = 128  # Number of features in the hidden state
output_size = 1  # Output size for binary classification
num_layers = 1  # Number of recurrent layers
num_epochs = 10  # Number of epochs
batch_size = 32  # Batch size
learning_rate = 0.001  # Learning rate

# Define the RNN model
class SentimentRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SentimentRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)  # Fully connected layer for classification
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary classification

    def forward(self, x):
        rnn_out, _ = self.rnn(x)  # rnn_out shape: (batch_size, sequence_length, hidden_size)
        last_hidden = rnn_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        output = self.fc(last_hidden)  # Shape: (batch_size, output_size)
        return self.sigmoid(output)  # Sigmoid for binary output

# Load data
# Assuming selected_columns is the DataFrame with 'sentiment' and 'embeddings'
# Convert DataFrame to tensors

data = pd.read_pickle('data/review_data_embeddings.pkl')

X = torch.tensor(data['embeddings'].tolist(), dtype=torch.float32)  # Shape: (num_samples, max_length, input_size)
y = torch.tensor(data['sentiment'].tolist(), dtype=torch.float32).view(-1, 1)  # Shape: (num_samples, 1)

# Create a DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Split data into training and validation sets (80/20 split)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = SentimentRNN(input_size, hidden_size, output_size, num_layers)
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)  # Model prediction
        loss = criterion(outputs, labels)  # Compute loss

        # Backward pass and optimization
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters

        if (i + 1) % 100 == 0:  # Print every 100 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Evaluation function
def evaluate(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    total, correct = 0, 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
            outputs = model(inputs)  # Model prediction
            predicted = (outputs >= 0.5).float()  # Convert probabilities to binary output
            total += labels.size(0)  # Total samples
            correct += (predicted == labels).sum().item()  # Count correct predictions

    accuracy = correct / total * 100  # Calculate accuracy
    return accuracy

# Evaluate on the validation set
val_accuracy = evaluate(model, val_loader)
print(f'Validation Accuracy: {val_accuracy:.2f}%')

# Save the model
torch.save(model.state_dict(), 'sentiment_rnn_model.pth')
