# LoL 드래프트 승부 예측 — 전통적 ML vs LLM 일반지식

**롤(League of Legends) 드래프트(양 팀 챔피언 10개)만 보고 승부를 맞힐 수 있을까?**
전통적 머신러닝(피처엔지니어링 + 16개 분류기 + SHAP)과 LLM(BERT 파인튜닝, Qwen3-32B 제로샷)을
동일한 train/val/test 분할에서 비교하고, 챔피언 임베딩을 UMAP 2D로 시각화한다.

> 핵심 결론: **챔피언 조합만으로는 신호가 매우 약하다.** 전통 ML은 AUC ≈ 0.54–0.57로
> 우연(0.50)보다 조금 높을 뿐이고, LLM의 일반 지식(Qwen3-32B 제로샷)은 사실상 우연 수준이며
> 심하게 과신(log-loss ≈ 1.1)한다. 승부의 대부분은 드래프트가 아니라 플레이어 실력에서 결정된다.

> 📄 정식 논문 형식 정리: **[docs/PAPER.md](docs/PAPER.md)** (Abstract · Methods · Results · Discussion)

---

## 1. 데이터

원본: 같은 저장소의 자매 프로젝트 [`../embedding-analysis/`](../embedding-analysis/)가 생성한
라벨 인코딩 CSV 3종 (`embedding-analysis/data/processed/`). 동일한 `label_encoder.pkl`
(`embedding-analysis/weights/`, 챔피언 0–191, 실제 등장 171)로 인코딩.

| 데이터셋 | 경기 수 | 블루 승률 | 설명 |
|---|---|---|---|
| `pro` | 10,008 | 0.533 | 프로 e스포츠 (Oracle's Elixir 2025), 24개 패치 |
| `solo_all` | 137,146 | 0.494 | 솔로랭크 전체 티어 |
| `solo_high` | 2,566 | 0.513 | 챌린저 |

- **입력**: `blue_p1..p5`, `red_p1..p5` (챔피언 ID), **타깃**: `result` (1 = 블루 승)
- 중복 드래프트가 거의 없어 암기가 불가능 → 순수 일반화 성능을 본다.
- 보조 자산: `champion.json`(DDragon: 태그·스탯 17차원 + 역할), `embedding_init.pt`(사전학습 챔피언 임베딩 [192,32]).

분할은 타깃 기준 **stratified 70/15/15**(시드 42)로 고정 — 모든 방법이 **같은 test 행**을 본다.

---

## 2. 방법

### 2.1 전통적 ML ([`src/train_classical.py`](src/train_classical.py))

**표현 4종** ([`src/features.py`](src/features.py))
- `bag` — Bag-of-Champions: 길이 192 부호 멀티핫 (블루 +1 / 레드 −1). 챔피언별 "강함"을 선형으로 학습. **주력**
- `meta` — DDragon 메타데이터 팀 조합 (탱커 수·총 AD 등, 51차원)
- `emb` — 사전학습 임베딩 팀 평균 (96차원)
- `combo` — 위 3개 결합 (339차원)

**분류기 16종** (5개 계열)
- 선형: Logistic Regression, Ridge, SGD-log, ElasticNet, Linear-SVM
- 거리/확률: kNN, RBF-SVM, Gaussian NB
- 트리 앙상블: RandomForest, ExtraTrees, GradientBoosting, HistGB, AdaBoost, XGBoost, LightGBM
- 신경망: MLP
- 기준선: 다수 클래스 / 블루-사이드 prior

> RBF-SVM·kNN은 비용이 O(n²)라 대용량(solo_all)에서 학습 표본을 12,000으로 캡(`n_train_used` 컬럼에 기록).

### 2.2 SHAP ([`src/shap_analysis.py`](src/shap_analysis.py))
XGBoost + Bag-of-Champions에 TreeExplainer 적용. 피처 = 챔피언이므로 SHAP 값이 곧
"그 챔피언이 블루 승률을 밀어올린/내린 정도" → 데이터로 학습한 챔피언 영향력 랭킹.

### 2.3 BERT 파인튜닝 ([`src/train_bert.py`](src/train_bert.py))
드래프트를 `"Blue team: ... . Red team: ... ."` 텍스트로 변환 → `bert-base-uncased` 2-클래스 분류 파인튜닝.
**데이터로부터 학습**하는 텍스트 모델.

### 2.4 Qwen3-32B 제로샷 ([`src/llm_zeroshot.py`](src/llm_zeroshot.py))
파인튜닝 없음. 두 팀을 보여주고 다음 토큰이 "Blue"인지 "Red"인지의 로짓을 비교 →
확률 → **test 전체** 평가. **LLM의 일반 지식만** 사용.

> ⚠️ 이 Blackwell GPU 머신은 GPU↔GPU **P2P 텐서 복사가 0으로 깨지는** 버그가 있어
> `device_map="auto"` 분할 시 hidden state가 전부 0이 된다(원인 추적은 git 로그 참고).
> 따라서 Qwen은 **단일 GPU에 8-bit로 적재**(≈35GB)해 우회한다.

---

## 3. 결과

*(수치는 `results/metrics/all_methods.csv` · `classical.csv` 기준. `scripts/run_all.sh`로 재현)*

### 3.1 방법별 비교 (test split)

| 데이터셋 | 방법 | Acc | AUC | LogLoss | Brier |
|---|---|---|---|---|---|
| pro | **전통ML best: extratrees+bag** | 0.555 | 0.561 | 0.686 | 0.246 |
| pro | BERT (파인튜닝) | 0.533 | 0.505 | 0.691 | 0.249 |
| pro | Qwen3-32B (제로샷) | 0.530 | 0.484 | 1.049 | 0.349 |
| pro | 기준선: 블루-사이드 prior | 0.533 | 0.500 | 0.691 | 0.249 |
| solo_all | **전통ML best: logreg+combo** | 0.525 | 0.536 | 0.691 | 0.249 |
| solo_all | BERT (파인튜닝) | 0.495 | 0.503 | 0.693 | 0.250 |
| solo_all | Qwen3-32B (제로샷) | 0.495 | 0.503 | 1.156 | 0.374 |
| solo_all | 기준선: 블루-사이드 prior | 0.505 | 0.500 | 0.693 | 0.250 |
| solo_high | **전통ML best: logreg+combo** | 0.558 | 0.570 | 0.722 | 0.261 |
| solo_high | BERT (파인튜닝) | 0.514 | 0.482 | 0.694 | 0.250 |
| solo_high | Qwen3-32B (제로샷) | 0.522 | 0.550 | 1.121 | 0.357 |
| solo_high | 기준선: 블루-사이드 prior | 0.512 | 0.500 | 0.693 | 0.250 |

### 3.2 전통 ML 계열별 최고 (solo_all, test AUC 기준)

| 계열 | 최고 모델+표현 | Acc | AUC |
|---|---|---|---|
| linear | logreg+combo | 0.525 | 0.536 |
| distance | svm_rbf+combo | 0.513 | 0.526 |
| probabilistic | gnb+meta | 0.516 | 0.524 |
| tree | xgb+bag | 0.523 | 0.532 |
| neural | mlp+combo | 0.521 | 0.530 |

### 그림 ([`figures/`](figures/))
- `compare_accuracy.png` — 방법 계열별 최고 정확도·AUC (데이터셋 × 계열)
- `roc_<ds>.png`, `calibration_<ds>.png`, `confusion_<ds>.png` — 진단
- `shap_summary_<ds>.png` — 챔피언별 SHAP 기여도 (beeswarm)
- `umap_champion_embeddings.png` — 챔피언 임베딩 4종(init/bag/BERT/Qwen) UMAP, 역할 색상
- `umap_draft_separability_solo_all.png` — 드래프트 수준 UMAP(승/패) → 클래스가 거의 겹침

### 해석
- 가장 큰 신호는 **블루 사이드 이점**(특히 pro 0.533)과 소수의 강/약 챔피언.
- 구조화된 `bag` one-hot이 텍스트(BERT)·메타데이터·임베딩보다 일관되게 우위 — BERT는 챔피언 이름
  텍스트에서 매치업 정보를 끌어내지 못하고 거의 다수 클래스(AUC≈0.50)에 머문다.
- Qwen3-32B 제로샷은 **랭킹은 우연 근처, 확률은 심하게 과신**(log-loss≈1.1) — 일반 LLM 지식이
  메타 승률을 수치로 인코딩하지 못함을 보여준다.

---

## 4. 재현

```bash
cd prediction-benchmark
bash scripts/run_all.sh        # 전통 ML → SHAP → BERT → Qwen → 시각화
```
데이터는 자매 프로젝트 `embedding-analysis/`에서 자동 참조한다(별도 클론 불필요).
환경: 프로젝트 venv (`/projects/PSALM/thisaint/venv`, torch 2.11+cu128, transformers 5.9),
GPU(BERT·Qwen). 추가 패키지는 [`requirements.txt`](requirements.txt).

## 5. 구조
```
src/        config · champion_meta · data · features · utils · draft_text
            train_classical · shap_analysis · train_bert · llm_zeroshot
            viz_umap · viz_results
results/    metrics/*.csv · shap/*.csv · preds/*.npz · embeddings/*.npy
figures/    *.png
scripts/    run_all.sh
```
