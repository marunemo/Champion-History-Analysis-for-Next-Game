"""Fine-tune BERT to predict blue win from the verbalised draft.

Each draft becomes "Blue team: ... . Red team: ... ." and bert-base-uncased is
fine-tuned as a 2-class sequence classifier. This tests whether a text model,
*learning from the match data*, can extract draft signal — the counterpart to
Qwen's zero-shot general knowledge.

Also extracts a fine-tuned per-champion embedding (mean-pooled last hidden state
of the champion name) for the UMAP visualisation.

Outputs:
  results/preds/bert_<ds>.npz            y_true, p_blue on test split
  results/metrics/bert.csv               per-dataset test metrics
  results/embeddings/bert_champ_<ds>.npy [N_CHAMPIONS, hidden]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import (DATASETS, TARGET, BERT_MODEL, METRICS_DIR, PREDS_DIR,
                    EMB_DIR, SEED, N_CHAMPIONS)
from data import load_dataset, make_splits
from draft_text import draft_to_text, names
from utils import compute_metrics, fmt_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 64


def encode_texts(tokenizer, texts):
    return tokenizer(texts, padding="max_length", truncation=True,
                     max_length=MAX_LEN, return_tensors="pt")


def make_loader(tokenizer, df, batch, shuffle):
    enc = encode_texts(tokenizer, [draft_to_text(r) for _, r in df.iterrows()])
    y = torch.tensor(df[TARGET].values, dtype=torch.long)
    ds = TensorDataset(enc["input_ids"], enc["attention_mask"], y)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    ps, ys = [], []
    for ids, mask, y in loader:
        out = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
        p = torch.softmax(out.logits.float(), -1)[:, 1].cpu().numpy()
        ps.append(p); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


@torch.no_grad()
def champion_embeddings(model, tokenizer):
    """Mean-pooled last-hidden-state of each champion name."""
    model.eval()
    embs = []
    base = model.bert if hasattr(model, "bert") else model.base_model
    for nm in names():
        enc = tokenizer(nm, return_tensors="pt", truncation=True, max_length=8).to(DEVICE)
        out = base(**enc)
        h = out.last_hidden_state[0]                      # [T, H]
        m = enc["attention_mask"][0].unsqueeze(-1).float()
        emb = (h * m).sum(0) / m.sum()
        embs.append(emb.float().cpu().numpy())
    return np.stack(embs)


def train_one(name, epochs, batch, lr):
    torch.manual_seed(SEED)
    df = load_dataset(name)
    sp = make_splits(df)
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=2).to(DEVICE)

    tr = make_loader(tok, sp["train"], batch, True)
    va = make_loader(tok, sp["val"], batch, False)
    te = make_loader(tok, sp["test"], batch, False)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))
    lossf = torch.nn.CrossEntropyLoss()

    best_auc, best_state = -1, None
    for ep in range(epochs):
        model.train()
        for ids, mask, y in tr:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
                out = model(input_ids=ids.to(DEVICE), attention_mask=mask.to(DEVICE))
                loss = lossf(out.logits.float(), y.to(DEVICE))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        yv, pv = predict(model, va)
        mv = compute_metrics(yv, pv)
        print(f"  [{name}] epoch {ep+1}/{epochs} val {fmt_metrics(mv)}")
        if mv["roc_auc"] > best_auc:
            best_auc = mv["roc_auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    yt, pt = predict(model, te)
    mt = compute_metrics(yt, pt)
    print(f"  [{name}] TEST {fmt_metrics(mt)}")
    np.savez(PREDS_DIR / f"bert_{name}.npz", y_true=yt, p_blue=pt)
    np.save(EMB_DIR / f"bert_champ_{name}.npy", champion_embeddings(model, tok))
    return dict(dataset=name, model="bert", **mt)


def run(epochs, batch, lr, only=None):
    rows = []
    for name in DATASETS:
        if only and name != only:
            continue
        print(f"\n===== BERT fine-tune: {name} =====")
        rows.append(train_one(name, epochs, batch, lr))
    pd.DataFrame(rows).to_csv(METRICS_DIR / "bert.csv", index=False)
    print("Saved -> results/metrics/bert.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    run(a.epochs, a.batch, a.lr, a.only)
