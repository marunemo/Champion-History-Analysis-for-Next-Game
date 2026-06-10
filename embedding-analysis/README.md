# Champion Draft Embedding Analysis for LoL

League of Legends 챔피언 드래프트(10-pick)로 승패를 예측하는 모델을 학습하고,
학습된 **챔피언 임베딩 공간을 분석**하여 시너지, 메타 이탈, 티어별 드래프트 철학을 해석하는 프로젝트.

> 📄 분석 내용을 논문 형태로 풀어 쓴 글: **[docs/PAPER.md](docs/PAPER.md)**

## Project Structure

```
project/
├── data/
│   ├── raw/                       # 원본 데이터 (champion.json만 추적, 나머지 gitignore)
│   └── processed/                 # 전처리 완료 CSV
│       ├── df_pro_enc.csv         (10,008행 — 프로 경기)
│       ├── df_solo_all_enc.csv    (137,146행 — 솔로랭크 전체)
│       ├── df_solo_high_enc.csv   (2,566행 — GM+Challenger)
│       └── df_solo_low_enc.csv    (13,984행 — Iron~Silver)
├── weights/
│   ├── embedding_init.pt          DDragon 초기화 weight (192, 32)
│   ├── embedding_B_learned.pt     Model B 학습 임베딩 (transfer learning init)
│   └── label_encoder.pkl          sklearn LabelEncoder (192 챔피언)
├── src/
│   ├── preprocess.py              data/raw/ → data/processed/ + embedding_init.pt
│   ├── model.py                   DraftEmbeddingFFNN + DraftEmbeddingCNN
│   ├── dataset.py                 DraftDataset + swap augmentation
│   ├── train.py                   학습 루프 (Adam, BCE, early stopping)
│   └── analyze.py                 임베딩 분석 6종
├── outputs/
│   ├── models/
│   │   ├── baseline/              {ffnn|cnn}_{A|B|C|D}.pt (8개)
│   │   └── transfer/              {ffnn|cnn}_{A|C|D}_tl.pt
│   ├── figures/                   시각화 PNG (tsne / pca / archetype / delta_shift)
│   ├── training_report.md         학습 결과 요약 (baseline + transfer)
│   ├── analysis_report.md         임베딩 분석 해석 리포트
│   └── experiment_design.md       실험 설계 문서
├── docs/
│   └── PAPER.md                   분석 내용을 논문 형태로 풀어 쓴 글 (outputs/ 리포트 기반)
└── requirements.txt
```

## Model Architectures

```
DraftEmbeddingFFNN                          DraftEmbeddingCNN
  Embedding(192, 32) ← DDragon init          Embedding(192, 32) ← DDragon init
  ↓ concat(blue5 + red5 = 320)               ↓ stack(10, 32) → permute(32, 10)
  Linear(320, 256) → ReLU → Drop(0.3)        Conv1d(32,64,k=3) → ReLU
  Linear(256, 64) → ReLU                     Conv1d(64,128,k=3) → ReLU
  Linear(64, 1) → Sigmoid                    AdaptiveAvgPool1d(1) → Linear(128,64)
                                              → ReLU → Drop(0.3) → Linear(64,1) → Sigmoid
```

## Experiment Setup

| Model | Dataset | Rows | FFNN Acc | CNN Acc |
|-------|---------|------|----------|---------|
| A | Pro matches (Oracle's Elixir 2025) | 10,008 | 53.1% | 52.8% |
| B | Solo Queue all ranks | 137,146 | 53.2% | 53.1% |
| C | Solo Queue GM+Challenger | 2,566 | 52.7% | 51.4% |
| D | Solo Queue Iron~Silver | 13,984 | 51.8% | 51.3% |

- Blue/Red swap augmentation (train data 2x, side-bias removed)
- Adam lr=1e-3, BCELoss, max 30 epochs, early stopping patience=5

## Embedding Analysis (6 types × 2 architectures × datasets)

1. **t-SNE** — Before/After 비교, DDragon role 태그 6색
2. **Archetype Clustering** — 5-pick mean-pool → KMeans(k=7), 조합 원형 분류
3. **Delta Weight** — L2 norm(W_after - W_before), DDragon 대비 메타 이탈 상위 20
4. **PCA** — 주성분 축 해석 + DDragon stat 상관 분석
5. **PCA Annotated** — PC1-PC2 / PC1-PC3 축 극단 챔피언 라벨링
6. **t-SNE Migration** — Before→After 임베딩 이동 화살표 (상위 이탈 챔피언)

## Key Findings

- **FFNN ≈ CNN**: 성능 차이 < 0.003. 드래프트에서 중요한 것은 "누가 있느냐"이지 위치 순서가 아님
- **PC1 (37~46%)**: 교전 접근성 — 원거리 지원 vs 근접 교전
- **PC2 (16~18%)**: 솔로 에이전시 — DDragon으로 설명 불가, 순수 게임 데이터에서 학습된 축
- **고티어 vs 저티어**: C는 어쌔신-다이버 중심, D는 유틸리티 챔피언 중심 — 티어별 드래프트 철학의 근본적 차이
- **최대 메타 이탈**: Azir, K'Sante, Skarner — 25.19+ 패치 정체성 변동 반영
- **Swap augmentation**: Blue side bias 제거로 Model C val_acc 49.6% → 52.7%

상세 분석은 [outputs/analysis_report.md](outputs/analysis_report.md) 참고.

## Data Sources

| Data | Source |
|------|--------|
| Pro matches | Oracle's Elixir 2025 |
| Solo Queue | nathansmallcalder (Kaggle), Patch 25.19+ |
| Champion attributes | [DDragon 15.19.1](https://ddragon.leagueoflegends.com/cdn/15.19.1/data/en_US/champion.json) |

## Quick Start

> 모든 명령은 이 폴더(`embedding-analysis/`)를 작업 디렉터리로 두고 실행합니다.

```bash
pip install -r requirements.txt

# (Optional) Regenerate processed CSV + embedding init from data/raw/
python -m src.preprocess

# Train (FFNN)
python -m src.train --data data/processed/df_pro_enc.csv --name A --arch ffnn
python -m src.train --data data/processed/df_solo_all_enc.csv --name B --arch ffnn
python -m src.train --data data/processed/df_solo_high_enc.csv --name C --arch ffnn
python -m src.train --data data/processed/df_solo_low_enc.csv --name D --arch ffnn

# Train (CNN)
python -m src.train --data data/processed/df_pro_enc.csv --name A --arch cnn
python -m src.train --data data/processed/df_solo_all_enc.csv --name B --arch cnn
python -m src.train --data data/processed/df_solo_high_enc.csv --name C --arch cnn
python -m src.train --data data/processed/df_solo_low_enc.csv --name D --arch cnn

# Transfer learning (B-learned embedding → A/C/D fine-tune)
python -m src.train --data data/processed/df_pro_enc.csv --name A_tl --arch ffnn --init-weights weights/embedding_B_learned.pt
python -m src.train --data data/processed/df_solo_high_enc.csv --name C_tl --arch ffnn --init-weights weights/embedding_B_learned.pt
python -m src.train --data data/processed/df_solo_low_enc.csv --name D_tl --arch ffnn --init-weights weights/embedding_B_learned.pt

# Analyze (baseline + transfer)
python -m src.analyze --arch ffnn --models A B C D A_tl C_tl D_tl
python -m src.analyze --arch cnn --models A B C D A_tl C_tl D_tl
```
