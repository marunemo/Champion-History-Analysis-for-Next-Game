import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


PICK_COLS_BLUE = ["blue_p1", "blue_p2", "blue_p3", "blue_p4", "blue_p5"]
PICK_COLS_RED = ["red_p1", "red_p2", "red_p3", "red_p4", "red_p5"]


def _swap_augment(df):
    """Double data by swapping blue/red sides and flipping the label."""
    swapped = df.copy()
    for i in range(1, 6):
        swapped[f"blue_p{i}"] = df[f"red_p{i}"].values
        swapped[f"red_p{i}"] = df[f"blue_p{i}"].values
    swapped["result"] = 1 - df["result"].values
    return pd.concat([df, swapped], ignore_index=True)


class DraftDataset(Dataset):
    def __init__(self, df):
        self.blue = torch.tensor(df[PICK_COLS_BLUE].values, dtype=torch.long)
        self.red = torch.tensor(df[PICK_COLS_RED].values, dtype=torch.long)
        self.labels = torch.tensor(df["result"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.blue[idx], self.red[idx], self.labels[idx]


def get_loaders(csv_path, batch_size=256, val_ratio=0.2, seed=42, augment=True):
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Split BEFORE augmentation to prevent data leakage
    split = int(len(df) * (1 - val_ratio))
    train_df = df.iloc[:split]
    val_df = df.iloc[split:]

    if augment:
        train_df = _swap_augment(train_df)
        train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_loader = DataLoader(DraftDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(DraftDataset(val_df), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
