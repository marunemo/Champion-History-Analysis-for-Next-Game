"""Render a label-encoded draft row as natural-language text.

Shared by the BERT fine-tuning and the Qwen zero-shot scripts so both LLM
methods see the exact same verbalisation of a draft.
"""
from __future__ import annotations
from config import BLUE_COLS, RED_COLS, N_CHAMPIONS
from champion_meta import build_meta

_, _, _NAMES, _ = build_meta(N_CHAMPIONS)


def champ_name(cid: int) -> str:
    return _NAMES[int(cid)] if 0 <= int(cid) < len(_NAMES) else f"Champ{cid}"


def draft_to_text(row) -> str:
    blue = ", ".join(champ_name(row[c]) for c in BLUE_COLS)
    red = ", ".join(champ_name(row[c]) for c in RED_COLS)
    return f"Blue team: {blue}. Red team: {red}."


def names():
    return list(_NAMES)
