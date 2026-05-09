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
