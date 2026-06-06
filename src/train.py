import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.model import DraftEmbeddingFFNN, DraftEmbeddingCNN
from src.dataset import get_loaders

ARCH_MAP = {"ffnn": DraftEmbeddingFFNN, "cnn": DraftEmbeddingCNN}


def set_seed(seed):
    """Fix all RNGs for reproducible training (dropout, shuffle, CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for blue, red, labels in loader:
        blue, red, labels = blue.to(device), red.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(blue, red).squeeze(1)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    for blue, red, labels in loader:
        blue, red, labels = blue.to(device), red.to(device), labels.to(device)
        preds = model(blue, red).squeeze(1)
        total_loss += criterion(preds, labels).item() * labels.size(0)
        correct += ((preds >= 0.5).float() == labels).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--arch", default="ffnn", choices=["ffnn", "cnn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--init-weights", default="weights/embedding_init.pt",
                        help="Path to embedding init weights (default: DDragon)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.arch.upper()} {args.name}] device={device}, data={args.data}, augment={args.augment}, seed={args.seed}")

    train_loader, val_loader = get_loaders(
        args.data, batch_size=args.batch_size, augment=args.augment, seed=args.seed
    )

    ModelClass = ARCH_MAP[args.arch]
    model = ModelClass(init_weight_path=args.init_weights).to(device)
    w_before = torch.load(args.init_weights, weights_only=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    model_dir = "transfer" if args.name.endswith("_tl") else "baseline"
    save_path = f"outputs/models/{model_dir}/{args.arch}_{args.name}.pt"

    pbar = tqdm(range(1, args.epochs + 1), desc=f"{args.arch.upper()} {args.name}")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        pbar.set_postfix(
            tr_loss=f"{train_loss:.4f}",
            vl_loss=f"{val_loss:.4f}",
            vl_acc=f"{val_acc:.4f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "arch": args.arch,
                    "model_state_dict": model.state_dict(),
                    "w_before": w_before,
                    "best_val_loss": best_val_loss,
                    "best_val_acc": val_acc,
                    "epoch": epoch,
                },
                save_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    ckpt = torch.load(save_path, weights_only=False)
    print(
        f"\n[{args.arch.upper()} {args.name}] Best epoch {ckpt['epoch']}: "
        f"val_loss={ckpt['best_val_loss']:.4f}, val_acc={ckpt['best_val_acc']:.4f}"
    )


if __name__ == "__main__":
    main()
