import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter  # Import TensorBoard
import os

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
data = pd.read_csv("student_data.csv", delimiter=';')

# Map output to number
data['Output'] = data['Output'].apply(lambda x: 0 if x == 'Dropout' else 1)

# Prepare the input and output vectors
X = data.drop(columns=['Output']).values
y = data['Output'].values

# Normalize the input vector
X_scaled = StandardScaler().fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device).view(-1, 1)

# Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Define the neural network
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], 32)
        self.fc2 = nn.Linear(32, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Instantiate the model, define loss and optimizer
model = NeuralNetwork().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Set up TensorBoard writer
log_dir = os.path.join("logs", "fit")
writer = SummaryWriter(log_dir)

# Training the model
epochs = 25
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    # Log the loss to TensorBoard
    writer.add_scalar('Loss/train', loss.item(), epoch)

# Close the writer
writer.close()

# Evaluate the model
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    predicted = (test_outputs >= 0.5).float()
    accuracy = (predicted.eq(y_test_tensor).sum().item() / y_test_tensor.size(0))
    print(f'Accuracy: {accuracy:.4f}')

# Print model summary
print(model)
