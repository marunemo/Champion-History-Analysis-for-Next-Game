# Champion History Analysis for Next Game

> **드래프트(양 팀 챔피언 10개)만 보고 League of Legends 경기를 예측·해석할 수 있는가?**

하나의 질문을 두 각도에서 접근하는 두 개의 자매 프로젝트로 구성된다. 둘은 **같은 데이터셋과
같은 `label_encoder`(챔피언 0–191)** 를 공유한다.

| 프로젝트 | 묻는 것 | 방법 |
|---|---|---|
| [**embedding-analysis/**](embedding-analysis/) | 모델이 **무엇을 학습했나** (챔피언 임베딩 공간 해석) | FFNN / CNN 학습 + 전이학습, t-SNE · PCA · archetype · delta-weight · migration |
| [**prediction-benchmark/**](prediction-benchmark/) | **얼마나 맞히나** (방법론 계열 비교) | 전통 ML 16종 + SHAP vs BERT 파인튜닝 vs Qwen3-32B 제로샷, UMAP |

## 한 줄 결론

**챔피언 조합만으로는 신호가 매우 약하다.** 예측 성능은 우연(0.50)보다 조금 높은 AUC ≈ 0.54–0.57에
그치고, LLM의 일반 지식(Qwen3-32B 제로샷)은 사실상 우연 수준이며 심하게 과신한다. 그 약한 신호가
임베딩 공간에서 무엇을 의미하는지는 `embedding-analysis/`가 해석한다 — 교전 접근성·솔로 에이전시
같은 축, 티어별 드래프트 철학의 차이, 패치 메타 이탈(Azir·K'Sante·Skarner 등).

## 데이터 (공유)

라벨 인코딩된 드래프트 CSV와 사전학습 챔피언 자산은 `embedding-analysis/`가 생성하고,
`prediction-benchmark/`는 이를 그대로 참조한다(중복 없음).

```
embedding-analysis/data/processed/df_*.csv     라벨 인코딩 드래프트 (blue_p1..5, red_p1..5, result)
embedding-analysis/data/raw/champion.json      DDragon 메타데이터 (태그·스탯·역할)
embedding-analysis/weights/label_encoder.pkl   챔피언 id <-> 이름 (192종, 실제 등장 171)
embedding-analysis/weights/embedding_init.pt   DDragon 초기화 임베딩 [192, 32]
```

| 데이터셋 | 경기 수 | 출처 |
|---|---|---|
| pro | 10,008 | Oracle's Elixir 2025 (프로 e스포츠) |
| solo_all | 137,146 | 솔로랭크 전체 티어 (Kaggle, Patch 25.19+) |
| solo_high | 2,566 | GM+Challenger |
| solo_low | 13,984 | Iron~Silver (embedding-analysis 전용) |

## 구조

```
.
├── embedding-analysis/    임베딩 공간 해석 (src/ outputs/ data/ weights/) — 데이터 producer
└── prediction-benchmark/  예측 성능 비교 (src/ docs/ figures/ results/) — 데이터 consumer
```

각 프로젝트는 자기완결형이다 — 실행법·결과·의존성은 각 폴더의 README를 참고한다.
