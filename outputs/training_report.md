# Training Report — DraftEmbedding (FFNN & CNN)

## Environment
- PyTorch: 2.9.1+cu130
- Device: CUDA (GPU)
- DDragon init weight: (192, 32)

## Architectures

### FFNN (DraftEmbeddingFFNN)
```
Embedding(192, 32) → concat(blue 5 + red 5 = 320) → Linear(320,256) → ReLU → Dropout(0.3) → Linear(256,64) → ReLU → Linear(64,1) → Sigmoid
```

### CNN (DraftEmbeddingCNN)
```
Embedding(192, 32) → stack(10, 32) → Conv1d(32,64,k=3) → ReLU → Conv1d(64,128,k=3) → ReLU → AdaptiveAvgPool1d(1) → Linear(128,64) → ReLU → Dropout(0.3) → Linear(64,1) → Sigmoid
```

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
| Augmentation | Blue/Red swap (train 2x) |

## Results

| Model | Dataset | Rows | Arch | Best Epoch | Val Loss | Val Acc |
|-------|---------|------|------|-----------|----------|---------|
| A | Pro (Oracle's Elixir 2025) | 10,008 | FFNN | 1 | 0.6910 | 53.1% |
| A | Pro | 10,008 | CNN | 7 | 0.6915 | 52.8% |
| B | Solo Queue all | 137,146 | FFNN | 7 | 0.6897 | 53.2% |
| B | Solo Queue all | 137,146 | CNN | 7 | 0.6898 | 53.1% |
| C | Solo Queue GM+Chall | 2,566 | FFNN | 4 | 0.6923 | 52.7% |
| C | Solo Queue GM+Chall | 2,566 | CNN | 10 | 0.6929 | 51.4% |
| D | Solo Queue Iron~Silver | 13,984 | FFNN | 2 | 0.6921 | 51.8% |
| D | Solo Queue Iron~Silver | 13,984 | CNN | 8 | 0.6918 | 51.3% |

## Observations

### Swap Augmentation 효과
- 이전 실험(augmentation 없음) 대비 전체적으로 val_acc이 안정화됨
- 특히 Model C가 이전 49.6% → 52.7%로 개선 (Blue side 편향 제거 효과)

### FFNN vs CNN
- 두 아키텍처의 성능 차이가 거의 없음 (val_loss 차이 < 0.003)
- **해석**: 드래프트 내 "위치 순서"(p1~p5)는 승패에 유의미한 정보를 제공하지 않음
- CNN이 더 많은 epoch을 학습하는 경향 — Conv1d가 임베딩을 더 느리게 변형함

### 티어별 비교 (Model C vs D)
- 고티어(GM+Chall)와 저티어(Iron~Silver) 모두 비슷한 정확도 (~52%)
- 그러나 Archetype/임베딩 분석에서 메타 구조가 근본적으로 다름 (별도 분석 참고)

### 데이터량 효과
- Model B(137K)가 여전히 최저 val_loss — 데이터량이 임베딩 품질의 핵심 결정 요인

## Checkpoint Format
`outputs/models/{arch}_{model}.pt`:
- `arch`: "ffnn" 또는 "cnn"
- `model_state_dict`: 학습된 모델 가중치
- `w_before`: DDragon 초기화 임베딩 (Δweight 분석용)
- `best_val_loss`, `best_val_acc`, `epoch`
