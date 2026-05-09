# Embedding Analysis Report — DraftEmbedding (FFNN & CNN)

## 개요

네 모델(A: 프로, B: 솔로랭크 전체, C: 솔로랭크 고티어, D: 솔로랭크 저티어)을
두 아키텍처(FFNN, 1D-CNN)로 학습하여 총 8개 모델의 챔피언 임베딩을 분석하였다.
모든 학습에는 **Blue/Red swap augmentation**을 적용하여 side bias를 제거하였다.

---

## 실험 설계 변경 사항 (v2)

| 항목 | v1 | v2 |
|------|----|----|
| 모델 수 | 3 (A/B/C) | **4** (A/B/C/D) |
| 아키텍처 | FFNN only | **FFNN + 1D-CNN** |
| Augmentation | 없음 | **Blue/Red swap (train 2x)** |
| 분석 종류 | 5종 | **4종** (heatmap 제외 — 데이터 의존적이라 아키텍처 비교에 부적합) |
| 저티어 데이터 | 없음 | **Iron~Silver (13,984행)** |

---

## 1. t-SNE 클러스터 (Before/After)

### 관찰

- **Before (DDragon init)**: 모든 모델이 동일한 초기화에서 출발.
  DDragon tag·info·stats 17차원의 Xavier 투영. 역할 기반 클러스터가 이미 존재.
  - Marksman/Support 클러스터가 가장 선명, Fighter/Tank은 겹침

- **After**:
  - **FFNN B / CNN B (솔로 전체, 137K)**: 클러스터 경계 흐려짐. Fighter-Assassin-Mage 혼합.
    → "역할 구분"보다 "플레이 패턴 유사성"이 임베딩을 지배.
  - **FFNN A / CNN A (프로, 10K)**: DDragon 구조 대체로 유지. CNN이 FFNN보다 약간 더 변형
    (7 epoch vs 1 epoch — CNN이 더 천천히 학습하므로 더 많이 돌아감).
  - **FFNN D / CNN D (저티어, 14K)**: 저티어도 데이터량 덕분에 일부 임베딩 변형 발생.
    특히 Lux, Teemo, MissFortune 등 저티어 인기 챔피언 주변 밀집도 증가.
  - **C (고티어, 2.5K)**: 여전히 DDragon 초기화 지배적.

### FFNN vs CNN 비교
두 아키텍처의 t-SNE 구조가 매우 유사 → **임베딩 학습에 있어 아키텍처보다 데이터량이 지배적**.

---

## 2. Archetype 클러스터링 (KMeans k=7)

### 프로(A) vs 솔로(B) — 조합 정형화 차이

- **Model A**: 클러스터가 시각적으로 잘 분리됨. 프로씬의 정형화된 드래프트 반영.
  - "탱크 프론트라인 조합" (K'Sante/Braum/Sion/Alistar)
  - "AP 탑 + 이니시 조합" (Neeko/Vi/Taliyah/Rumble)
  - "밴 존 원거리 딜 조합" (Corki/Varus/Rell/Azir)

- **Model B**: 클러스터 경계 불명확. 137K 게임의 비정형적 드래프트.
  - "솔로 캐리형" (Akali/Yasuo/Yone/LeeSin) vs "유틸 원딜" (MissFortune/Jhin/Ashe/Nami)
    이 두 극단만 구분 가능.

### 고티어(C) vs 저티어(D) — 핵심 대비

- **Model C (GM+Chall)**: Ambessa/Zed/Akali/Qiyana 중심의 **어쌔신-다이버 조합**이 지배적.
  소수 데이터에서도 "솔로킬 특화" 패턴이 선명하게 드러남.

- **Model D (Iron~Silver)**: Lux/MissFortune/Teemo/Malphite/Morgana 중심.
  **간단한 조작 + 높은 유틸리티** 챔피언이 지배. 조합 원형이 "쉬운 챔피언 모음"으로 수렴.

→ **티어 간 드래프트 철학의 근본적 차이**: 고티어는 "개인기 극대화", 저티어는 "안정적 유틸리티".

---

## 3. Δweight — DDragon 대비 메타 이탈도

### FFNN 기준

| Model | Δ 크기 | Top 3 이탈 챔피언 | 해석 |
|-------|--------|-----------------|------|
| A (프로) | ~0.08 | Morgana, Jarvan IV, Poppy | 역할 이탈 (Mage→Support, 등) |
| B (솔로 전체) | **~1.3** | Azir, K'Sante, Skarner | 패치 기반 정체성 변동 |
| C (고티어) | ~0.08 | Ashe, Zeri, Sett | 고티어 특수 운용 |
| D (저티어) | ~0.08 | (DDragon 초기화 유지) | 데이터 부족으로 큰 변형 없음 |

### CNN vs FFNN
- CNN의 Δ 크기가 FFNN과 거의 동일한 분포 → 아키텍처가 임베딩 변형 패턴에 유의미한 영향을 주지 않음
- 이는 두 아키텍처가 **동일한 임베딩 레이어를 공유하는 구조**이기 때문에 예상된 결과

---

## 4. PCA 주성분 분석

### PC별 해석 (FFNN B 기준, 가장 많은 데이터로 학습)

#### PC1 (37.7%): "교전 접근성" (Engagement Range)
- attackrange(r=+0.80), attackdamage(r=-0.77), armor(r=-0.67)
- (+) Milio, Vel'Koz, Sona, Seraphine — 원거리 마법/지원
- (-) K'Sante, Darius, Nocturne, Jarvan IV — 근접 물리/돌진
- **드래프트의 가장 중요한 단일 축**: "우리 팀이 근접 교전팀인가, 원거리 견제팀인가"

#### PC2 (15.6%): "솔로 에이전시" (Solo Agency)
- DDragon stat과 직접 상관 약함 — **순수하게 게임 데이터에서 학습된 축**
- (+) Nilah, Fizz, Akali, LeBlanc — 솔로킬/로밍 특화
- (-) Ezreal, Corki, Varus, Yunara — 팀파이트 원딜
- 고티어(C)에서 이 축의 분산이 상대적으로 클 것으로 예상 — **고티어일수록 solo agency 중요**

#### PC3 (15.1%): "프론트라인 vs 캐리" (Frontline Identity)
- defense(r=+0.57), attack(r=-0.41)
- (+) Maokai, Malphite, Singed, Leona — 순수 탱크
- (-) Akshan, Azir, Twitch, Vayne — 자가생존 캐리
- Model B에서 PC2와 PC3의 분산이 거의 동일 (15.6% vs 15.1%)
  → 솔로랭크에서 "솔로 에이전시"와 "프론트라인 정체성"이 동등하게 중요

### PCA 요약

| PC | 해석 | Model A | Model B | Model C | Model D |
|----|------|---------|---------|---------|---------|
| 1 | 교전 접근성 | 46.1% | 37.7% | 46.1% | ~46% |
| 2 | 솔로 에이전시 | 18.3% | 15.6% | 18.3% | ~18% |
| 3 | 프론트라인 정체성 | 13.4% | 15.1% | 13.3% | ~13% |

- A/C/D는 DDragon 초기화 지배 → 거의 동일한 분산 구조
- **B만 PC1 비중 감소 + PC2-3 증가** → 데이터가 충분할 때 새로운 의미 축이 학습됨

---

## 모델 간 비교 종합

| 관점 | A (프로) | B (솔로 전체) | C (고티어) | D (저티어) |
|------|---------|-------------|----------|----------|
| 데이터 | 10K | 137K | 2.5K | 14K |
| FFNN val_acc | 53.1% | 53.2% | 52.7% | 51.8% |
| CNN val_acc | 52.8% | 53.1% | 51.4% | 51.3% |
| Δweight | 작음 | **큼** | 작음 | 작음 |
| Archetype 분리 | 명확 | 뭉개짐 | 어쌔신 집중 | 유틸 집중 |
| 드래프트 특성 | 정형화·전략적 | 비정형·개인선호 | 솔로킬 특화 | 쉬운 챔프 수렴 |

---

## 핵심 발견 (v2 업데이트)

1. **Swap augmentation으로 Blue side bias 제거**: Model C의 val_acc가 49.6% → 52.7%로 개선.
   모든 모델에서 학습이 안정화됨.

2. **FFNN ≈ CNN**: 두 아키텍처의 성능 차이가 val_loss 기준 0.003 미만.
   드래프트 내 위치 순서(p1~p5)는 승패에 유의미한 정보를 제공하지 않음.
   → **LoL 드래프트에서 중요한 것은 "누가 있느냐"이지 "어느 위치에 있느냐"가 아님**.

3. **고티어 vs 저티어 드래프트 철학**: C는 어쌔신-다이버 중심, D는 유틸리티 챔피언 중심.
   티어가 올라갈수록 "솔로 에이전시"가 높은 챔피언이 선호됨.

4. **데이터량이 임베딩 품질의 결정적 요인**: B(137K)만이 DDragon 초기화를 탈피.
   D(14K)도 A(10K)보다 데이터가 많지만, 저티어의 비정형적 드래프트로 인해 의미 있는 임베딩 변형이 제한적.

5. **PCA가 드러낸 3대 잠재 축**:
   - 교전 접근성 (37~46%) — 원거리/마법 vs 근접/물리
   - 솔로 에이전시 (16~18%) — 솔로킬 vs 팀 의존
   - 프론트라인 정체성 (13~15%) — 탱크 vs 자가생존 캐리

---

## 한계 및 향후 과제

- Model A/C의 학습 부족: 데이터 증강만으로는 한계, transfer learning 검토 필요
- PCA 주성분 해석은 상관 기반 추론 (인과관계 아님)
- Co-occurrence 분석을 Blue-Red 교차로 확장하면 **카운터 관계** 정량화 가능
- 솔로랭크 데이터에서 게임 시간(duration) 정보를 활용하면 "조기 항복 vs 후반 역전" 조합 구분 가능
