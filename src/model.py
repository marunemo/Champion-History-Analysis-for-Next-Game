import torch
import torch.nn as nn


class DraftEmbeddingFFNN(nn.Module):
    def __init__(self, num_champions=192, embed_dim=32, init_weight_path=None):
        super().__init__()
        self.embed = nn.Embedding(num_champions, embed_dim)

        if init_weight_path is not None:
            init_weight = torch.load(init_weight_path, weights_only=True)
            self.embed.weight = nn.Parameter(init_weight.clone())

        self.fc = nn.Sequential(
            nn.Linear(embed_dim * 5 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, blue_picks, red_picks):
        blue_emb = self.embed(blue_picks).view(-1, 5 * self.embed.embedding_dim)
        red_emb = self.embed(red_picks).view(-1, 5 * self.embed.embedding_dim)
        return self.fc(torch.cat([blue_emb, red_emb], dim=1))


class DraftEmbeddingCNN(nn.Module):
    """1D-CNN over the 10-pick sequence.

    Input: (batch, 5) blue + (batch, 5) red → stacked as (batch, 10, embed_dim)
    Conv1D treats embed_dim as channels, sequence length = 10.
    """

    def __init__(self, num_champions=192, embed_dim=32, init_weight_path=None):
        super().__init__()
        self.embed = nn.Embedding(num_champions, embed_dim)

        if init_weight_path is not None:
            init_weight = torch.load(init_weight_path, weights_only=True)
            self.embed.weight = nn.Parameter(init_weight.clone())

        # Conv1D: (batch, channels=embed_dim, seq_len=10)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (batch, 128, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, blue_picks, red_picks):
        # (batch, 5, embed_dim) each → concat → (batch, 10, embed_dim)
        blue_emb = self.embed(blue_picks)
        red_emb = self.embed(red_picks)
        x = torch.cat([blue_emb, red_emb], dim=1)  # (batch, 10, embed_dim)
        x = x.permute(0, 2, 1)  # (batch, embed_dim, 10) for Conv1d
        x = self.conv(x).squeeze(-1)  # (batch, 128)
        return self.fc(x)
