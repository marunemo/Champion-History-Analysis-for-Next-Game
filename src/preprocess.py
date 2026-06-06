"""
LOL Draft Embedding — Preprocessing Pipeline

DDragon 챔피언 특성 + Oracle's Elixir(프로) + Kaggle 솔로랭크 데이터를
모델 학습용 CSV + 임베딩 초기화 weight로 변환한다.

Data Sources:
    1. Riot DDragon 15.19.1 — champion.json
       https://ddragon.leagueoflegends.com/cdn/15.19.1/data/en_US/champion.json
    2. Oracle's Elixir — 2025 LoL Esports Match Data
       https://oracleselixir.com/tools/downloads
    3. Nathan Smallcalder — LoL Matches Patch 25.19+ (Kaggle)
       https://www.kaggle.com/datasets/nathansmallcalder/lol-match-history-and-summoner-data-80k-matches

Usage:
    python src/preprocess.py          # data/raw/ → data/processed/, weights/
"""

import csv
import io
import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ── 경로 설정 ──────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "processed")
WEIGHT_DIR = os.path.join(ROOT, "weights")

OE_PATH = os.path.join(RAW_DIR, "2025_LoL_esports_match_data_from_OraclesElixir.csv")
SR_PATH = os.path.join(RAW_DIR, "kaggle_matches_25.19+")
MATCH_TBL = os.path.join(SR_PATH, "MatchTbl.csv")
TEAM_MATCH_TBL = os.path.join(SR_PATH, "TeamMatchTbl.csv")
CHAMPION_TBL = os.path.join(SR_PATH, "ChampionTbl.csv")
RANK_TBL = os.path.join(SR_PATH, "RankTbl.csv")
CHAMPION_JSON = os.path.join(RAW_DIR, "champion.json")

# ── 상수 ──────────────────────────────────────────
DDRAGON_VERSION = "15.19.1"
ALL_TAGS = ["Fighter", "Tank", "Mage", "Assassin", "Marksman", "Support"]
STAT_COLS = ["hp", "armor", "spellblock", "attackdamage",
             "attackspeed", "movespeed", "hpregen"]
EMBED_DIM = 32

# DDragon 미지원 챔피언 (패치 이후 추가)
UNKNOWN_CHAMPIONS = {"Zaahen"}

# 매치 데이터 표시명 → DDragon key 매핑 (21건)
DDRAGON_NAME_MAP = {
    "Aurelion Sol":   "AurelionSol",
    "Bel'Veth":       "Belveth",
    "Cho'Gath":       "Chogath",
    "Dr. Mundo":      "DrMundo",
    "Jarvan IV":      "JarvanIV",
    "K'Sante":        "KSante",
    "Kai'Sa":         "Kaisa",
    "Kha'Zix":        "Khazix",
    "Kog'Maw":        "KogMaw",
    "LeBlanc":        "Leblanc",
    "Lee Sin":        "LeeSin",
    "Master Yi":      "MasterYi",
    "Miss Fortune":   "MissFortune",
    "Nunu & Willump": "Nunu",
    "Rek'Sai":        "RekSai",
    "Renata Glasc":   "Renata",
    "Tahm Kench":     "TahmKench",
    "Twisted Fate":   "TwistedFate",
    "Vel'Koz":        "Velkoz",
    "Wukong":         "MonkeyKing",
    "Xin Zhao":       "XinZhao",
}

# ── 로그 ──────────────────────────────────────────
_log_buffer = io.StringIO()


def log(msg: str = ""):
    _log_buffer.write(msg + "\n")
    print(msg)


def flush_log():
    path = os.path.join(OUT_DIR, "preprocessing_log.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_log_buffer.getvalue())


# ── Pipeline 1: DDragon → Champion Feature Matrix ──
def fetch_ddragon_features() -> pd.DataFrame:
    """DDragon champion.json → 챔피언별 특성 DataFrame (192, 17)"""
    log("=" * 60)
    log("STEP 1: DDragon Feature Matrix")

    with open(CHAMPION_JSON, "r", encoding="utf-8") as f:
        raw = json.load(f)["data"]

    records = []
    for name, champ in raw.items():
        tag_vec = {f"tag_{t}": int(t in champ["tags"]) for t in ALL_TAGS}
        info = champ["info"]
        info_vec = {
            "info_attack":     info["attack"] / 10.0,
            "info_defense":    info["defense"] / 10.0,
            "info_magic":      info["magic"] / 10.0,
            "info_difficulty": info["difficulty"] / 10.0,
        }
        stats = champ["stats"]
        stat_vec = {col: stats.get(col, 0.0) for col in STAT_COLS}
        records.append({"champion_name": name, **tag_vec, **info_vec, **stat_vec})

    df = pd.DataFrame(records).set_index("champion_name")
    scaler = MinMaxScaler()
    df[STAT_COLS] = scaler.fit_transform(df[STAT_COLS])

    log(f"  챔피언 수  : {len(df)}")
    log(f"  feature 차원: {df.shape[1]}  (tag×6, info×4, stats×7)")
    log()
    return df


# ── Pipeline 2: Oracle's Elixir (프로) ──
def load_oracles_elixir() -> pd.DataFrame:
    """Oracle's Elixir 2025 CSV → blue_p1~5, red_p1~5, result"""
    log("=" * 60)
    log("STEP 2: Oracle's Elixir 2025")

    df = pd.read_csv(OE_PATH, low_memory=False)
    team = df[df["position"] == "team"].copy()
    log(f"  team row 수: {len(team):,}")

    pick_cols = [f"pick{i}" for i in range(1, 6)]
    keep = ["gameid", "side", "result", *pick_cols]
    team = team[keep].dropna(subset=pick_cols)
    for col in pick_cols:
        team[col] = team[col].str.strip()

    blue = (team[team["side"] == "Blue"]
            .rename(columns={f"pick{i}": f"blue_p{i}" for i in range(1, 6)})
            [["gameid", "blue_p1", "blue_p2", "blue_p3", "blue_p4", "blue_p5", "result"]])
    red = (team[team["side"] == "Red"]
           .rename(columns={f"pick{i}": f"red_p{i}" for i in range(1, 6)})
           [["gameid", "red_p1", "red_p2", "red_p3", "red_p4", "red_p5"]])

    merged = blue.merge(red, on="gameid", how="inner")
    log(f"  병합 게임 수: {len(merged):,}")
    log()
    return merged


# ── Pipeline 3: 솔로랭크 ──
def load_solo_rank(rank_name_filter=None, rank_fk_filter=None,
                   label="전체") -> pd.DataFrame:
    """
    Kaggle 솔로랭크 데이터 → blue_p1~5, red_p1~5, result

    Args:
        rank_name_filter: RankName 필터 (e.g. ["Grandmaster", "Challenger"])
        rank_fk_filter:   RankFk 필터 (e.g. [1, 2, 3] for Iron~Silver)
        label:            로그용 라벨
    """
    log("=" * 60)
    log(f"STEP 3: Solo Rank [{label}]")

    match = pd.read_csv(MATCH_TBL)
    tms = pd.read_csv(TEAM_MATCH_TBL)
    champ = pd.read_csv(CHAMPION_TBL)
    rank_map = {}
    with open(RANK_TBL, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rank_map[int(row["RankId"])] = row["RankName"]

    id2name = dict(zip(champ["ChampionId"], champ["ChampionName"]))

    match = match[match["QueueType"] == "CLASSIC"].copy()
    match["RankName"] = match["RankFk"].map(rank_map)

    if rank_name_filter is not None:
        match = match[match["RankName"].isin(rank_name_filter)]
    if rank_fk_filter is not None:
        match = match[match["RankFk"].isin(rank_fk_filter)]

    valid_ids = set(match["MatchId"])
    tms = tms[tms["MatchFk"].isin(valid_ids)].copy()

    b_cols = [f"B{i}Champ" for i in range(1, 6)]
    r_cols = [f"R{i}Champ" for i in range(1, 6)]
    for col in b_cols + r_cols:
        tms[col] = tms[col].map(id2name)
    tms = tms.dropna(subset=b_cols + r_cols)

    tms["result"] = tms["BlueWin"].astype(int)
    tms = tms.rename(columns={
        **{f"B{i}Champ": f"blue_p{i}" for i in range(1, 6)},
        **{f"R{i}Champ": f"red_p{i}" for i in range(1, 6)},
    })

    pick_cols = [f"blue_p{i}" for i in range(1, 6)] + \
                [f"red_p{i}" for i in range(1, 6)]
    tms = tms[pick_cols + ["result"]]

    log(f"  최종 게임 수: {len(tms):,}")
    log()
    return tms


# ── Pipeline 4: LabelEncoder ──
def build_champion_encoder(*dfs: pd.DataFrame) -> LabelEncoder:
    """여러 DataFrame의 챔피언명을 통합하여 LabelEncoder 생성"""
    log("=" * 60)
    log("STEP 4: Champion LabelEncoder")

    pick_cols = [f"blue_p{i}" for i in range(1, 6)] + \
                [f"red_p{i}" for i in range(1, 6)]

    all_names = pd.concat([
        pd.Series(d[pick_cols].values.flatten()) for d in dfs
    ]).dropna()

    unique = set(all_names.unique()) - UNKNOWN_CHAMPIONS
    le = LabelEncoder()
    le.fit(sorted(unique))

    log(f"  통합 챔피언 풀: {len(le.classes_)}")
    log()
    return le


def encode_draft_df(df: pd.DataFrame, le: LabelEncoder,
                    label: str = "") -> pd.DataFrame:
    """champion name → champion ID 변환, 미등록 챔피언 행 제거"""
    pick_cols = [f"blue_p{i}" for i in range(1, 6)] + \
                [f"red_p{i}" for i in range(1, 6)]
    known = set(le.classes_)
    mask = df[pick_cols].apply(lambda col: col.isin(known)).all(axis=1)
    df = df[mask].copy()

    for col in pick_cols:
        df[col] = le.transform(df[col])

    log(f"  [{label}] 인코딩 후: {len(df):,}행")
    return df


# ── Pipeline 5: Embedding Init Weight ──
def build_embedding_init(feat_df: pd.DataFrame,
                         le: LabelEncoder) -> torch.Tensor:
    """DDragon feature (17d) → Xavier projection (32d)"""
    log("=" * 60)
    log("STEP 5: Embedding Init Weight")

    num_champs = len(le.classes_)
    feat_dim = feat_df.shape[1]

    ordered = np.zeros((num_champs, feat_dim), dtype=np.float32)
    for idx, name in enumerate(le.classes_):
        ddragon_key = DDRAGON_NAME_MAP.get(name, name)
        if ddragon_key in feat_df.index:
            ordered[idx] = feat_df.loc[ddragon_key].values
        elif name in feat_df.index:
            ordered[idx] = feat_df.loc[name].values

    feat_tensor = torch.tensor(ordered)
    proj = nn.Linear(feat_dim, EMBED_DIM, bias=False)
    nn.init.xavier_uniform_(proj.weight)
    with torch.no_grad():
        weight = proj(feat_tensor)

    log(f"  shape: {tuple(weight.shape)}")
    log()
    return weight


# ── Main ──────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(WEIGHT_DIR, exist_ok=True)

    log("LOL Draft Embedding — Preprocessing Pipeline v2")
    log(f"DDragon Version: {DDRAGON_VERSION}")
    log()

    # Step 1
    feat_df = fetch_ddragon_features()

    # Step 2
    df_pro = load_oracles_elixir()

    # Step 3
    df_solo_all = load_solo_rank(label="전체")
    df_solo_high = load_solo_rank(
        rank_name_filter=["Grandmaster", "Challenger"],
        label="GM+Challenger")
    df_solo_low = load_solo_rank(
        rank_fk_filter=[1, 2, 3],
        label="Iron~Silver")

    # Step 4
    le = build_champion_encoder(df_pro, df_solo_all)
    log("인코딩:")
    df_pro_enc = encode_draft_df(df_pro, le, "Pro")
    df_solo_all_enc = encode_draft_df(df_solo_all, le, "Solo All")
    df_solo_high_enc = encode_draft_df(df_solo_high, le, "Solo High")
    df_solo_low_enc = encode_draft_df(df_solo_low, le, "Solo Low")
    log()

    # Step 5
    init_weight = build_embedding_init(feat_df, le)

    # Export
    log("=" * 60)
    log("EXPORT")
    exports = {
        "df_pro_enc.csv": df_pro_enc,
        "df_solo_all_enc.csv": df_solo_all_enc,
        "df_solo_high_enc.csv": df_solo_high_enc,
        "df_solo_low_enc.csv": df_solo_low_enc,
    }
    for name, df in exports.items():
        path = os.path.join(OUT_DIR, name)
        df.to_csv(path, index=False)
        log(f"  {name}: {len(df):,}행")

    torch.save(init_weight, os.path.join(WEIGHT_DIR, "embedding_init.pt"))
    log(f"  embedding_init.pt: {tuple(init_weight.shape)}")

    with open(os.path.join(WEIGHT_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    log(f"  label_encoder.pkl: {len(le.classes_)} 챔피언")

    flush_log()
    log()
    log("완료.")


if __name__ == "__main__":
    main()
