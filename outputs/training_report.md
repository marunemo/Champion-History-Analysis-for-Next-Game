# Training Report — DraftEmbeddingFFNN

## Environment
- PyTorch: 2.9.1+cu130
- Device: CUDA (GPU)
- Architecture: FFNN (embed_dim=32 × 10 picks → 256 → 64 → 1, Sigmoid)
- DDragon init weight: (192, 32)

## Hyperparameters
| Param | Value |
|-------|-------|
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Loss | BCELoss |
| Max epochs | 30 |
| Early stopping patience | 5 |
| Batch size | 256 |
| Train/Val split | 80/20 |
| Dropout | 0.3 |

## Results

| Model | Dataset | Rows | Best Epoch | Val Loss | Val Accuracy |
|-------|---------|------|-----------|----------|-------------|
| A | df_pro_enc (프로 경기) | 10,008 | 2 | 0.6920 | 53.15% |
| B | df_solo_all_enc (솔로랭크 전체) | 137,146 | 9 | 0.6900 | 53.21% |
| C | df_solo_high_enc (솔로랭크 GM+Challenger) | 2,566 | 5 | 0.6930 | 49.61% |

## Observations
- 전체적으로 ~53% 수준의 정확도 — 드래프트(5v5 픽) 정보만으로 승패를 예측하는 것은 원래 어려운 문제
- Model B가 가장 많은 데이터(137K)를 가지면서 가장 낮은 val_loss 기록
- Model C는 데이터가 2.5K로 매우 적어 학습이 불안정 (val_acc < 50%)
- 본 프로젝트의 핵심은 정확도가 아닌 **임베딩 공간 분석** (t-SNE, co-occurrence, archetype, Δweight)

## Checkpoint Format
각 모델 체크포인트(`outputs/models/model_{A|B|C}.pt`)에 포함된 항목:
- `model_state_dict`: 학습된 모델 가중치
- `w_before`: DDragon 초기화 임베딩 (Δweight 분석용)
- `best_val_loss`: 최적 검증 손실
- `best_val_acc`: 최적 검증 정확도
- `epoch`: 최적 에포크 번호
