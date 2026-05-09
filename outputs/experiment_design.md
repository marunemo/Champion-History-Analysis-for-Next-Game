# 실험 설계 문서 — LoL 드래프트 임베딩 분석

## 1. 프로젝트 목표

League of Legends 챔피언 픽 조합(10-pick draft)으로 승패를 예측하는 모델을 구축하되,
**정확도 경쟁이 아닌 챔피언 임베딩 공간의 의미론적 분석**이 핵심이다.

구체적으로:
- DDragon 설계 속성(tag, info, stats)으로 초기화된 임베딩이 실제 게임 데이터로 학습되면서 어떻게 변형되는지 분석
- 프로/솔로랭크/고티어/저티어 환경에서 챔피언의 "기능적 역할"이 어떻게 달라지는지 비교
- FFNN과 CNN 아키텍처 비교를 통해 드래프트 내 위치 순서의 정보 가치를 검증

---

## 2. 데이터 파이프라인

### 2.1 원본 데이터

| 데이터 | 출처 | 전처리 | 저장 위치 |
|--------|------|--------|----------|
| 프로 경기 | Oracle's Elixir 2025 | position="team" 행 필터, gameid 기준 Blue/Red 병합 | `data/raw/` |
| 솔로랭크 | Kaggle: nathansmallcalder | QueueType=CLASSIC 필터, RankTbl로 티어 매핑 | `data/raw/kaggle_matches_25.19+/` |
| 챔피언 속성 | Riot DDragon 15.19.1 | champion.json에서 tags, info, stats 추출 | `data/raw/champion.json` |

#### 데이터 출처 상세

1. **Riot Games Data Dragon (DDragon)**
   - URL: `https://ddragon.leagueoflegends.com/cdn/15.19.1/data/en_US/champion.json`
   - 패치 15.19.1 기준 챔피언 메타데이터 (태그, 능력치 정보, 기본 스탯)
   - 취득: `wget -O champion.json "https://ddragon.leagueoflegends.com/cdn/15.19.1/data/en_US/champion.json"`

2. **Oracle's Elixir — 2025 LoL Esports Match Data**
   - 다운로드 페이지: https://oracleselixir.com/tools/downloads
   - Google Drive: https://drive.google.com/drive/folders/1gLSw0RLjBbtaNy0dgnGQDAZOHIgCe-HH
   - 2025 시즌 프로 대회 전 경기 데이터 (픽, 밴, KDA, 오브젝트 등)

3. **Nathan Smallcalder — League of Legends(LoL) Matches Patch 25.19+**
   - Kaggle: https://www.kaggle.com/datasets/nathansmallcalder/lol-match-history-and-summoner-data-80k-matches
   - 솔로랭크 80K+ 매치 데이터 (MatchTbl, TeamMatchTbl, ChampionTbl, RankTbl 등)

### 2.2 챔피언 특성 행렬 (17차원)

`embedding_init.pt` 생성에 사용되는 DDragon 특성 구성:

```
feat_tensor (192, 17) = [tags_onehot(6) | info_scaled(4) | stats_minmax(7)]
```

| 구간 | 차원 | 컬럼 | 정규화 |
|------|------|------|--------|
| tags | 6 | Fighter, Tank, Mage, Assassin, Marksman, Support | one-hot (0 or 1) |
| info | 4 | attack, defense, magic, difficulty | ÷ 10.0 → [0, 1] |
| stats | 7 | hp, armor, spellblock, attackdamage, attackspeed, movespeed, hpregen | MinMaxScaler → [0, 1] |

**포함되지 않은 주요 특성:**
- `attackrange` — DDragon stats에 존재하나 STAT_COLS에서 제외됨
- 스킬별 `range`, `cooldown`, `cost` — per-champion 상세 엔드포인트에만 존재
- `mp`, `crit`, `perlevel` 계열 — base stat만 사용

### 2.3 임베딩 초기화 (embedding_init.pt)

```python
feat_tensor = torch.tensor(ordered)              # (192, 17), 정규화 완료
proj = nn.Linear(17, 32, bias=False)              # Xavier uniform 초기화
nn.init.xavier_uniform_(proj.weight)
with torch.no_grad():
    weight = proj(feat_tensor)                    # (192, 32)
torch.save(weight, "embedding_init.pt")
```

**핵심 특성:**
1. **투영 행렬은 무작위**: Xavier uniform 난수. DDragon 특성이 유사한 챔피언은 초기 임베딩에서도 가깝지만, 32개 축의 의미는 임의적. "순수 난수보다 나은 시작점"이지 의미론적 공간이 아님.
2. **비재현성**: `torch.manual_seed` 미설정. 재실행 시 다른 초기화 생성 (현재 파일은 고정 사용).
3. **학습 아님**: `torch.no_grad()` 내에서 단순 선형 투영. 역전파 없음.

**이것이 분석에 미치는 영향:**
- t-SNE "Before" 플롯은 DDragon tag 기반 클러스터를 보여주되, 축 자체는 임의적
- Δweight 분석에서 이탈이 큰 챔피언 = DDragon이 포착 못한 메타 역할을 모델이 학습한 것
- **`attackrange`가 입력에 없는데 PC1이 이를 강하게 포착(r=+0.82)** → 모델이 드래프트 승패 데이터로부터 사거리 정보를 간접 학습했다는 증거

### 2.4 LabelEncoder

- 프로 + 솔로랭크 전체 데이터에서 등장하는 모든 챔피언명을 통합
- DDragon 미지원 챔피언(Zaahen 등 패치 이후 추가) 제거
- 표시명→DDragon key 매핑 21건 (e.g., "Jarvan IV"→"JarvanIV", "Kai'Sa"→"Kaisa")
- 최종: **192 챔피언**, 알파벳 순 정렬

### 2.5 전처리 완료 데이터

| 파일 | 행수 | 용도 | 필터 조건 |
|------|------|------|-----------|
| df_pro_enc.csv | 10,008 | Model A | Oracle's Elixir 2025, position=team |
| df_solo_all_enc.csv | 137,146 | Model B | QueueType=CLASSIC, 전체 티어 |
| df_solo_high_enc.csv | 2,566 | Model C | RankName ∈ {Grandmaster, Challenger} |
| df_solo_low_enc.csv | 13,984 | Model D | rank_fk ∈ {1(Iron), 2(Bronze), 3(Silver)} |

공통 스키마: `blue_p1~p5`, `red_p1~p5` (LabelEncoder int), `result` (1=Blue win, 0=Red win)

---

## 3. 모델 아키텍처

### 3.1 DraftEmbeddingFFNN (Baseline)

```
Embedding(192, 32)  ← embedding_init.pt 로드, requires_grad=True
  ↓
blue_picks(batch, 5) → embed → flatten(batch, 160)
red_picks(batch, 5)  → embed → flatten(batch, 160)
  ↓ concat → (batch, 320)
Linear(320, 256) → ReLU → Dropout(0.3)
Linear(256, 64) → ReLU
Linear(64, 1) → Sigmoid
```

- 10개 픽의 임베딩을 **단순 concat** → 위치 순서 정보 무시
- 파라미터 수: 192×32 + 320×256 + 256×64 + 64×1 ≈ 105K

### 3.2 DraftEmbeddingCNN

```
Embedding(192, 32)  ← 동일 초기화
  ↓
blue(batch, 5, 32) + red(batch, 5, 32) → stack → (batch, 10, 32)
  ↓ permute → (batch, 32, 10)  [channels=embed_dim, seq=10]
Conv1d(32, 64, kernel=3, padding=1) → ReLU
Conv1d(64, 128, kernel=3, padding=1) → ReLU
AdaptiveAvgPool1d(1) → (batch, 128)
  ↓
Linear(128, 64) → ReLU → Dropout(0.3)
Linear(64, 1) → Sigmoid
```

- 10개 픽을 **1D 시퀀스로 취급** → 인접 위치 간 지역적 상호작용 포착 가능
- Blue 5픽 + Red 5픽이 연속 배치되므로 blue_p5↔red_p1 경계에서 팀 간 상호작용 포착 기대
- 파라미터 수: 192×32 + Conv1d 계열 + FC ≈ 23K (FFNN의 ~22%)

### 3.3 공통 설계 결정

| 항목 | 선택 | 이유 |
|------|------|------|
| 임베딩 초기화 | DDragon → Xavier 투영 | 순수 난수보다 구조적 시작점 제공 |
| 임베딩 학습 | requires_grad=True | fine-tune으로 메타 정보 반영 |
| 출력 활성화 | Sigmoid + BCELoss | 이진 분류 (Blue win or not) |
| Dropout | 0.3 | 과적합 방지 (특히 소규모 데이터 A/C) |

---

## 4. 학습 설정

### 4.1 하이퍼파라미터

| Param | Value | 비고 |
|-------|-------|------|
| Optimizer | Adam | lr=1e-3 |
| Loss | BCELoss | 이진 분류 |
| Max Epochs | 30 | |
| Early Stopping | patience=5 | val_loss 기준 |
| Batch Size | 256 | |
| Train/Val Split | 80/20 | shuffle, seed=42 |

### 4.2 Blue/Red Swap Augmentation

```python
# 원본: (blue_picks, red_picks, result=1)  → Blue 승
# 변환: (red_picks, blue_picks, result=0)  → Red 승 (= Blue 패)
```

**설계 의도:**
- 프로 데이터의 Blue side 승률 53.3% 편향 제거
- 모델이 "Blue에 있기 때문에 유리하다"를 학습하는 것을 방지
- Train 데이터만 2배 증강 (val은 원본 유지 — 데이터 누수 방지)

**효과:**
- Model C (가장 소규모, 2.5K): val_acc 49.6% → 52.7% (+3.1%p)
- 모든 모델에서 학습 안정화 확인

### 4.3 학습 결과

| Model | Dataset | Rows | FFNN Epoch/Acc | CNN Epoch/Acc |
|-------|---------|------|----------------|---------------|
| A | Pro (Oracle's Elixir) | 10,008 | 1 / 53.1% | 7 / 52.8% |
| B | Solo Queue All | 137,146 | 7 / 53.2% | 7 / 53.1% |
| C | Solo Queue GM+Chall | 2,566 | 4 / 52.7% | 10 / 51.4% |
| D | Solo Queue Iron~Silver | 13,984 | 2 / 51.8% | 8 / 51.3% |

**관찰:**
- 전체적으로 ~52% 수준. 드래프트 정보만으로 승패 예측은 본질적으로 어려운 문제.
- **FFNN ≈ CNN** (val_loss 차이 < 0.003). 드래프트 내 위치 순서는 승패에 무관.
- CNN이 FFNN보다 더 많은 epoch을 소화 — Conv1d가 임베딩을 더 느리게 변형.
- **데이터량이 학습 깊이를 결정**: B(137K)만 7+ epoch, A/D(10~14K)는 1~2 epoch에서 early stop.

### 4.4 체크포인트 구조

`outputs/models/{ffnn|cnn}_{A|B|C|D}.pt`:
```python
{
    "arch": "ffnn" | "cnn",
    "model_state_dict": ...,
    "w_before": embedding_init.pt 원본 (Δweight 분석용),
    "best_val_loss": float,
    "best_val_acc": float,
    "epoch": int,
}
```

---

## 5. 임베딩 분석

### 5.1 분석 1: t-SNE 클러스터 (Before/After)

**방법**: 학습 전(DDragon init) / 학습 후 임베딩 (192, 32)에 t-SNE 적용, DDragon role 태그 6색으로 색상 구분.

**핵심 발견:**
- Before: Marksman/Support 클러스터 선명, Fighter/Tank 혼합 (DDragon 스탯 유사성)
- After(B, 137K): 클러스터 경계 흐려짐 → "역할 구분"보다 "플레이 패턴 유사성"이 지배
- After(A/C/D): DDragon 구조 대체로 유지 — 데이터 부족으로 초기화 탈피 실패

### 5.2 분석 2: Archetype 클러스터링 (KMeans k=7)

**방법**: Blue팀 5픽의 임베딩을 mean-pool → (N_games, 32) → KMeans(k=7). t-SNE로 시각화.

**핵심 발견:**
- **프로(A)**: 정형화된 조합 원형 — "탱크 프론트라인", "AP 탑 이니시", "원거리 딜" 등 명확 분리
- **솔로랭크(B)**: 비정형적 드래프트로 클러스터 경계 불명확
- **고티어(C)**: 어쌔신-다이버 조합이 지배 (Ambessa/Zed/Akali/Qiyana)
- **저티어(D)**: 유틸리티 챔피언 수렴 (Lux/MissFortune/Teemo/Malphite)
- → **티어 간 드래프트 철학의 근본적 차이**: 고티어는 "개인기 극대화", 저티어는 "안정적 유틸리티"

### 5.3 분석 3: Δweight (DDragon 대비 메타 이탈도)

**방법**: `Δ = W_after - W_before`, L2 norm 계산. 상위 20 챔피언 막대 그래프.

**핵심 발견:**
- Model B에서만 유의미한 이탈 (Δ norm ~1.3, 다른 모델의 15~20배)
- Top 3: **Azir(1.75)**, K'Sante(1.60), Skarner(1.57) — 25.19+ 패치 정체성 변동 반영
- Azir: DDragon Mage 태그이나 솔로랭크에서 원거리 DPS+딜탱 하이브리드로 운용
- K'Sante: DDragon Tank이나 실제로는 탑 브루저

### 5.4 분석 4: PCA 주성분 분석

**방법**: 학습된 임베딩 (192, 32)에 PCA 적용. 주성분과 DDragon stats 간 Pearson 상관 계산.

**핵심 발견:**

| PC | 분산 비율 | 해석 | DDragon 상관 |
|----|----------|------|-------------|
| 1 | 37~46% | **교전 접근성** (원거리 지원 vs 근접 교전) | attackrange: **+0.82**, attackdamage: -0.77 |
| 2 | 16~18% | **솔로 에이전시** (솔로킬 vs 팀 의존) | DDragon과 약상관 — 순수 게임 데이터 학습 축 |
| 3 | 13~15% | **프론트라인 정체성** (탱크 vs 자가생존 캐리) | defense: +0.57, attack: -0.41 |

**특기 사항 — attackrange의 간접 학습:**
`attackrange`는 임베딩 초기화의 17차원 입력에 **포함되지 않았다** (STAT_COLS에 없음).
그럼에도 학습된 임베딩의 PC1이 attackrange와 r=+0.82의 강한 상관을 보인다.

이는 두 가지 경로로 설명 가능:
1. DDragon 초기화 단계에서 tag one-hot(Marksman=원거리)과 info(attack/magic)가 사거리 정보를 부분적으로 인코딩
2. 학습 단계에서 드래프트 승패 패턴이 "원거리 챔피언 조합 vs 근접 챔피언 조합"의 승률 차이를 반영

→ 모델이 **명시적으로 주어지지 않은 게임 메커니즘(사거리)을 자율적으로 발견**했다는 해석이 가능하며,
이것이 임베딩 분석의 핵심 가치이다.

---

## 6. 실험 비교축 정리

### 6.1 데이터 축 (Model A / B / C / D)

| | A (프로) | B (솔로 전체) | C (고티어) | D (저티어) |
|---|---------|-------------|----------|----------|
| 행수 | 10K | 137K | 2.5K | 14K |
| Δweight | 작음 | **큼** | 작음 | 작음 |
| Archetype | 정형 | 비정형 | 어쌔신 집중 | 유틸 집중 |
| 드래프트 특성 | 전략적·팀 단위 | 개인 선호 | 솔로킬 특화 | 쉬운 챔프 수렴 |

### 6.2 아키텍처 축 (FFNN vs CNN)

| 관점 | FFNN | CNN |
|------|------|-----|
| 입력 처리 | 10픽 concat (320) | 10픽 1D 시퀀스 (Conv1d) |
| 위치 순서 | 무시 | 지역적 상호작용 포착 |
| 성능 | ≈ | ≈ (Δloss < 0.003) |
| 학습 속도 | 빠름 (적은 epoch) | 느림 (더 많은 epoch) |
| 파라미터 | ~105K | ~23K |

**결론: 위치 순서가 무의미** — p1~p5 배치가 승패에 영향을 주지 않음. LoL 드래프트에서 중요한 것은 "누가 있느냐"이지 "어느 위치에 있느냐"가 아님.

단, 이 데이터의 p1~p5가 실제 드래프트 순서(밴픽 오더)가 아닌 단순 포지션 나열일 가능성이 높으므로, "드래프트 순서가 무의미하다"는 결론은 아님에 유의.

### 6.3 증강 축 (Swap Augmentation)

| | 증강 전 (v1) | 증강 후 (v2) |
|---|-------------|-------------|
| Blue side bias | 있음 (53.3%) | **제거됨** |
| Model C acc | 49.6% | **52.7%** |
| 학습 안정성 | 불안정 | 안정 |

---

## 7. 한계 및 향후 과제

1. **데이터 불균형**: B(137K)만 충분하고, A(10K)/C(2.5K)/D(14K)는 192차원 임베딩을 재학습하기에 부족.
   Transfer learning (B 임베딩으로 초기화 → A/C/D fine-tune) 검토 가능.

2. **초기화 비재현성**: `torch.manual_seed` 미설정. 현재 파일 고정 사용으로 실험 결과에 영향 없으나,
   향후 재생성 시 시드 고정 필요.

3. **특성 제한**: attackrange, 스킬 사거리/쿨다운 등 미포함. 단, 모델이 이를 간접 학습한 것은 확인됨.

4. **PCA 해석의 한계**: 상관 기반 추론이므로 인과관계가 아닌 연관성.

5. **카운터 관계 미분석**: Blue-Red 교차 co-occurrence로 카운터 쌍 정량화 가능.

6. **p1~p5 순서의 의미 불확실**: 프로 데이터는 드래프트 순서, 솔로랭크는 포지션 나열일 가능성.
   이 차이가 CNN의 상대적 성능에 영향을 미쳤을 수 있음.
