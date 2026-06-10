"""Qwen3-32B zero-shot draft win prediction using *general knowledge only*.

No fine-tuning. We prompt the model with the two drafts and read off the
probability it assigns to the next token being "Blue" vs "Red" (single forward
pass per draft, batched). This isolates whether the LLM's pretrained knowledge
of champions encodes which composition wins.

Evaluated on the FULL test split of every dataset.

Also extracts a per-champion Qwen embedding (mean-pooled last hidden state of
the champion name) for the UMAP comparison.

Outputs:
  results/preds/qwen_<ds>.npz             y_true, p_blue on test split
  results/metrics/qwen.csv                per-dataset test metrics
  results/embeddings/qwen_champ.npy       [N_CHAMPIONS, hidden]
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig)

from config import (DATASETS, TARGET, QWEN_MODEL, METRICS_DIR, PREDS_DIR,
                    EMB_DIR, N_CHAMPIONS)
from data import load_dataset, make_splits
from draft_text import draft_to_text, names
from utils import compute_metrics, fmt_metrics

SYSTEM = ("You are a League of Legends draft analyst. Given both teams' champion "
          "picks, decide which side is more likely to win based on champion "
          "strength, matchups and team synergy.")
USER_TMPL = ("{draft}\n\nConsidering champion matchups and synergies, which side "
             "is more likely to win this game? Answer with exactly one word: "
             "Blue or Red.")


def build_prompt(tok, draft_text):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(draft=draft_text)}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    return text


def find_token_ids(tok, words):
    """Token ids whose decoded form (stripped, lowercased) starts with word."""
    ids = []
    for variant in words:
        for cand in (variant, " " + variant):
            enc = tok.encode(cand, add_special_tokens=False)
            if enc:
                ids.append(enc[0])
    return sorted(set(ids))


@torch.no_grad()
def score_split(model, tok, df, blue_ids, red_ids, batch=16, max_len=256):
    model.eval()
    texts = [build_prompt(tok, draft_to_text(r)) for _, r in df.iterrows()]
    probs = np.empty(len(texts), dtype=np.float32)
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to(model.device)
        out = model(**enc)
        # left-padding -> the final column is the last real token for every row
        logits = out.logits[:, -1, :].float()           # [b, V]
        lp_blue = torch.logsumexp(logits[:, blue_ids], dim=1)
        lp_red = torch.logsumexp(logits[:, red_ids], dim=1)
        p_blue = torch.softmax(torch.stack([lp_red, lp_blue], 1), 1)[:, 1]
        p_blue = torch.nan_to_num(p_blue, nan=0.5)
        probs[i:i + len(chunk)] = p_blue.cpu().numpy()
        if (i // batch) % 20 == 0:
            print(f"    scored {i+len(chunk)}/{len(texts)}", flush=True)
    return probs


@torch.no_grad()
def champion_embeddings(model, tok):
    embs = []
    for nm in names():
        enc = tok(nm, return_tensors="pt").to(model.device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0]              # [T, H]
        embs.append(h.mean(0).float().cpu().numpy())
    return np.stack(embs)


def run(batch, only=None, max_test=None):
    tok = AutoTokenizer.from_pretrained(QWEN_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    # NOTE: this Blackwell box has broken GPU<->GPU P2P copies (a tensor moved
    # cuda:0 -> cuda:1 comes out all zeros), so a device_map="auto" split across
    # both GPUs corrupts hidden states. We therefore load in 8-bit on a SINGLE
    # GPU (~35 GB, fits one 48 GB card) — no cross-GPU transfer.
    print("loading Qwen3-32B (8-bit, single GPU) ...", flush=True)
    bnb = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL, quantization_config=bnb, device_map={"": 0},
        attn_implementation="sdpa")
    blue_ids = find_token_ids(tok, ["Blue", "blue", "BLUE"])
    red_ids = find_token_ids(tok, ["Red", "red", "RED"])
    print("blue token ids:", blue_ids, "| red token ids:", red_ids)

    rows = []
    for name in DATASETS:
        if only and name != only:
            continue
        sp = make_splits(load_dataset(name))
        test = sp["test"]
        if max_test:
            test = test.iloc[:max_test]
        print(f"\n===== Qwen zero-shot: {name} (n_test={len(test)}) =====", flush=True)
        p = score_split(model, tok, test, blue_ids, red_ids, batch=batch)
        y = test[TARGET].values
        m = compute_metrics(y, p)
        print(f"  [{name}] {fmt_metrics(m)}", flush=True)
        np.savez(PREDS_DIR / f"qwen_{name}.npz", y_true=y, p_blue=p)
        rows.append(dict(dataset=name, model="qwen3-32b-zeroshot", **m))
        pd.DataFrame(rows).to_csv(METRICS_DIR / "qwen.csv", index=False)

    if not only:
        np.save(EMB_DIR / "qwen_champ.npy", champion_embeddings(model, tok))
    print("Saved -> results/metrics/qwen.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--only", default=None)
    ap.add_argument("--max_test", type=int, default=None)
    a = ap.parse_args()
    run(a.batch, a.only, a.max_test)
