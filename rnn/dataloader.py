import torch
from torch.utils.data import Dataset, DataLoader


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