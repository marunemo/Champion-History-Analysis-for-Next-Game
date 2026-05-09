# Champion Draft Embedding Analysis for LoL

League of Legends 챔피언 드래프트(10-pick)로 승패를 예측하는 FFNN 모델을 학습하고,
학습된 **챔피언 임베딩 공간을 분석**하여 시너지, 카운터, 메타 이탈을 해석하는 프로젝트.

## Project Structure

```
project/
├── data/                          # 전처리 완료 CSV
│   ├── df_pro_enc.csv             (10,008행 — 프로 경기)
│   ├── df_solo_all_enc.csv        (137,146행 — 솔로랭크 전체)
│   └── df_solo_high_enc.csv       (2,566행 — GM+Challenger)
├── weights/
│   ├── embedding_init.pt          DDragon 초기화 weight (192, 32)
│   └── label_encoder.pkl          sklearn LabelEncoder (192 챔피언)
├── src/
│   ├── model.py                   DraftEmbeddingFFNN
│   ├── dataset.py                 DraftDataset + DataLoader
│   ├── train.py                   학습 루프 (Adam, BCE, early stopping)
│   └── analyze.py                 임베딩 분석 5종
├── outputs/
│   ├── models/                    학습된 model_A/B/C.pt
│   ├── figures/                   시각화 PNG (15장)
│   ├── training_report.md         학습 결과 요약
│   └── analysis_report.md         임베딩 분석 해석 리포트
├── dataset/                       원본 데이터 + 전처리 노트북
├── notebooks/                     전처리 파이프라인 사본
└── requirements.txt
```

## Model Architecture

```
DraftEmbeddingFFNN
  Embedding(192, 32)  ← DDragon init (tag + info + stats → Xavier projection)
  ↓ blue_picks(5) + red_picks(5) → concat(320)
  Linear(320, 256) → ReLU → Dropout(0.3)
  Linear(256, 64)  → ReLU
  Linear(64, 1)    → Sigmoid
```

## Experiment Setup

| Model | Dataset | Rows | Best Epoch | Val Acc |
|-------|---------|------|-----------|---------|
| A | Pro matches (Oracle's Elixir 2025) | 10,008 | 2 | 53.2% |
| B | Solo Queue all ranks | 137,146 | 9 | 53.2% |
| C | Solo Queue GM+Challenger | 2,566 | 5 | 49.6% |

Hyperparameters: Adam lr=1e-3, BCELoss, max 30 epochs, early stopping patience=5, batch 256.

## Embedding Analysis (5 types)

1. **t-SNE** — Before/After 비교, DDragon role 태그 6색 (Fighter/Tank/Mage/Assassin/Marksman/Support)
2. **Co-occurrence vs Affinity** — 승리 조합 동반 등장 빈도 vs 임베딩 cosine similarity
3. **Archetype Clustering** — 5-pick mean-pool → KMeans(k=7)
4. **Delta Weight** — L2 norm(W_after - W_before), DDragon 대비 메타 이탈 상위 20
5. **PCA** — 주성분 축 해석 (교전 접근성, 솔로 에이전시, 자립도)

## Key Findings

- **PC1 (46%)**: 교전 접근성 축 — 원거리 마법(Support/Mage) vs 근접 물리(Tank/Fighter)
- **PC2 (18%)**: 솔로 에이전시 — 솔로킬 다이버(Akali/Fizz) vs 팀 의존형(Malphite/Kai'Sa)
- **PC3 (13%)**: 자립도 — 자가생존 원딜(Vayne/Quinn) vs 순수 CC 탱크(Maokai/Amumu)
- **최대 메타 이탈**: Azir(Δ=1.75), K'Sante(1.60), Skarner(1.57) — 25.19+ 패치 정체성 변동 반영
- **데이터량이 결정적**: 137K행의 Model B만 DDragon 초기화를 탈피하여 독립 임베딩 학습 성공

상세 분석은 [outputs/analysis_report.md](outputs/analysis_report.md) 참고.

## Data Sources

| Data | Source |
|------|--------|
| Pro matches | Oracle's Elixir 2025 |
| Solo Queue | nathansmallcalder (Kaggle), Patch 25.19+ |
| Champion attributes | [DDragon 15.19.1](https://ddragon.leagueoflegends.com/cdn/15.19.1/data/en_US/champion.json) |

## Quick Start

```bash
pip install -r requirements.txt

# Train
python -m src.train --data data/df_pro_enc.csv --name A
python -m src.train --data data/df_solo_all_enc.csv --name B
python -m src.train --data data/df_solo_high_enc.csv --name C

# Analyze
python -m src.analyze
```
