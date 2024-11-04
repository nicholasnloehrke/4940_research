import torch
from pathlib import Path

device = 

model = torch.jit.load(Path("models/test.pt"))
model.eval()

embedding = torch.randn((1, 100, 300))

all_hidden_states = []
all_labels = []

with torch.no_grad():
    for embeddings, labels in dataloader:
        embeddings = embeddings.to(device)
        labels = labels.to(device)

        # Forward pass through the LSTM
        outputs, (h_n, c_n) = model.rnn(embeddings.float())

        # Store hidden states for each word
        all_hidden_states.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        
        
# save_hidden_states.py
import torch
import torch.nn as nn
import pandas as pd
import argparse
from torch.utils.data import DataLoader
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_hidden_states(model, dataloader, output_file):
    model.eval()
    all_hidden_states = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            # Forward pass through the LSTM
            outputs, (h_n, c_n) = model.rnn(embeddings.float())

            # Store hidden states for each word
            all_hidden_states.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Save hidden states and labels
    np.savez_compressed(output_file, hidden_states=all_hidden_states, labels=all_labels)
    print(f"Hidden states saved to {output_file}")

def main(args):
    model = torch.jit.load(Path("models/test.pt"))
    model.eval()

    df = pd.read_pickle(args.embeddings_file)
    dataloader = create_dataloader(df, range(len(df)), config)
    
    save_hidden_states(model, dataloader, args.output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save hidden states from LSTM")
    parser.add_argument("embeddings_file", type=str, help="Input pickle file containing embeddings and sentiments")
    parser.add_argument("model_checkpoint", type=str, help="Path to trained LSTM model checkpoint")
    parser.add_argument("output_file", type=str, help="Output file to save hidden states")
    
    args = parser.parse_args()
    main(args)
