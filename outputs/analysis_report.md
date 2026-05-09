# Embedding Analysis Report — DraftEmbeddingFFNN

## 개요

세 모델(A: 프로, B: 솔로랭크 전체, C: 솔로랭크 고티어)에서 학습된 챔피언 임베딩 (192×32)을
DDragon 초기화 가중치와 비교하여 5종 분석을 수행하였다.

---

## 1. t-SNE 클러스터 (Before/After)

### 관찰

- **Before (DDragon init)**: 세 모델 모두 동일한 초기화에서 출발하므로 좌측 플롯은 동일하다.
  DDragon의 tag·info·stats 17차원을 Xavier 투영한 것으로, 이미 역할(role) 기반 클러스터가 존재한다.
  - Marksman(주황) 클러스터와 Support(청록) 클러스터가 가장 선명하게 분리됨
  - Fighter(빨강)와 Tank(파랑)은 겹치는 영역이 넓음 (DDragon 스탯 유사성)

- **After (학습 후)**:
  - **Model A (프로)**: 클러스터 구조가 Before와 유사하게 유지됨. Epoch 2에서 early stop되었으므로
    DDragon 초기화에서 크게 벗어나지 않았다. 프로 데이터 10K행은 192차원 임베딩을 대폭 변형하기에 부족.
  - **Model B (솔로 전체)**: 137K행의 데이터로 9 에폭 학습. 클러스터 간 경계가 흐려지고,
    특히 Fighter-Assassin-Mage가 혼합되는 영역이 넓어짐.
    → 솔로랭크에서는 역할 구분보다 "플레이 패턴의 유사성"이 임베딩을 지배함을 시사.
  - **Model C (솔로 고티어)**: 데이터 2.5K행, epoch 5. Model A와 거의 동일한 구조 유지.
    데이터 부족으로 DDragon 초기화를 탈피하지 못함.

### 해석

DDragon 태그와 학습된 임베딩의 불일치가 큰 챔피언은 **메타상 역할 이탈**(role drift)을 의미한다.
예: Corki(Marksman 태그)가 프로에서 미드 메이지처럼 운용되어 Mage 클러스터 근방으로 이동.

---

## 2. Co-occurrence vs Embedding Affinity 히트맵

### 관찰

- **Co-occurrence (왼쪽)**: 블루팀 승리 게임에서 5인 조합 내 동반 등장 빈도. 행 정규화 적용.
  - Model A(프로): 특정 쌍이 두드러짐 — Rell-Varus, Rumble-Corki, K'Sante-Alistar 등
    프로씬의 정형화된 드래프트 패턴이 반영됨.
  - Model B(솔로): 전체적으로 균일하게 분포. Nautilus-Kaisa가 눈에 띄는 조합 (서포터-원딜 시너지).
  - Model C(고티어): Karma-Tristana 조합이 강한 co-occurrence 보임.

- **Affinity (오른쪽)**: 학습된 임베딩의 cosine similarity.
  - 세 모델 모두 affinity 행렬이 전반적으로 양수(0.0~1.0)에 집중. 이는 임베딩 공간이
    아직 DDragon 초기화의 영향 아래 있어, 원점 방향 편향이 강하기 때문.
  - Model B에서 상대적으로 affinity 분산이 크며, Assassin-Tank 간 음의 affinity가 일부 관찰됨.

### 핵심 해석: 잠재 시너지 후보

**co-occurrence가 낮으나 affinity가 높은 쌍** = 데이터에서 자주 함께 쓰이진 않지만,
임베딩 공간에서는 유사하게 학습된 챔피언 → 잠재적 시너지 후보.
프로(Model A)에서 이 조건을 만족하는 쌍을 발굴하면 새로운 드래프트 전략 힌트가 될 수 있다.

---

## 3. Archetype 클러스터링 (KMeans k=7)

### 관찰

- **Model A (프로)**: 7개 클러스터가 시각적으로 잘 분리됨 (10K 게임, 비교적 정형화된 드래프트).
  - Cluster 0: Rumble/Corki/Varus/Rell/Azir — "밴 존 + 원거리 딜 조합"
  - Cluster 2: Braum/Yone/K'Sante/Sion/Alistar — "탱크 프론트라인 중심 조합"
  - Cluster 5: Neeko/Vi/Taliyah/Rumble/Rakan — "AP 탑 + 정글 이니시 조합"

- **Model B (솔로 전체)**: 137K 게임으로 클러스터가 뭉쳐 있고 경계가 불명확.
  솔로랭크에서는 조합 원형 구분이 희박함 — 드래프트가 비정형적이기 때문.
  - Cluster 0: Akali/Yasuo/Yone/LeeSin/Ambessa — "솔로 캐리형 어쌔신-파이터"
  - Cluster 4: MissFortune/Jhin/Smolder/Ashe/Nami — "유틸 원딜 + 인챈터"

- **Model C (고티어)**: A와 B의 중간. GM+챌린저는 프로보다 자유롭지만 솔로큐 전체보다 정형화.
  - Cluster 1/6: Ambessa/Yasuo/Akali/Zed/Viego — "하이 엘로 솔로킬 특화" 조합

### 해석

프로(A)에서 클러스터가 명확 → 드래프트 전략이 체계적.
솔로랭크(B)에서 클러스터가 뭉개짐 → 개인 선호와 무작위 드래프트 영향.
이 차이가 **"조합 난이도"** 지표로 해석 가능: 프로에서만 나타나는 클러스터 = 팀 조율이 필요한 고난도 전략.

---

## 4. Δweight — DDragon 대비 메타 이탈도

### 관찰

- **Model A (프로)**: 전체적으로 Δ norm이 0.07~0.09로 작음 (early stop epoch 2).
  - Top: Morgana(Support), Jarvan IV(Marksman→실제 정글), Poppy(Tank), Yunara(Marksman, 신챔)
  - Morgana가 1위인 것은 DDragon에서 Mage 태그이나 프로에서 서포터로 운용되는 역할 이탈 반영.

- **Model B (솔로 전체)**: Δ norm이 1.0~1.75로 Model A 대비 15~20배 큼. 9 에폭 + 137K 데이터로
  임베딩이 실질적으로 재학습됨.
  - Top: **Azir(1.75)**, K'Sante(1.60), Skarner(1.57), Veigar(1.37), Ezreal(1.35)
  - **Azir**: DDragon에서 Mage이나 솔로랭크에서는 "원거리 DPS + 딜탱" 하이브리드로 운용.
    높은 난이도 때문에 숙련자/비숙련자 간 임베딩 위치 차이가 극대화.
  - **K'Sante**: 25.S1 밸런스 패치로 역할이 크게 변동. DDragon의 Tank 스탯과 실제 "탑 브루저" 운용 간 괴리.
  - **Skarner**: 리워크 이후 정체성 변화.

- **Model C (고티어)**: Δ norm 0.06~0.10 (데이터 부족).
  - Top: Ashe, Zeri, Sett, Soraka, Syndra — GM 이상에서 특수한 방식으로 운용되는 챔피언.
  - **Soraka(Support)**가 높은 Δ를 보이는 것은, 고티어에서 Soraka의 솔로레인 운용이 반영되었을 가능성.

### 핵심 해석

Δweight 크기 = "DDragon 설계 의도와 실제 메타가 가장 괴리된 챔피언".
Model B의 Azir, K'Sante, Skarner가 압도적 → 이 챔피언들이 25.19+ 패치에서 정체성이 가장 크게 변형됨.

---

## 5. PCA 주성분 분석 (추가)

### Scree Plot

| PC | Model A (var%) | Model B (var%) | Model C (var%) |
|----|---------------|---------------|---------------|
| 1  | **46.1%**     | **37.7%**     | **46.1%**     |
| 2  | 18.3%         | 15.6%         | 18.3%         |
| 3  | 13.4%         | 15.1%         | 13.3%         |
| 4  | 6.1%          | 7.5%          | 6.1%          |
| 5  | 5.2%          | 6.2%          | 5.2%          |
| **누적 5** | **89.0%** | **82.1%** | **89.0%** |

Model A/C는 거의 동일한 분산 구조 (DDragon 초기화 지배적). Model B만 PC1 비중이 줄고 PC2-3이 올라감
→ 데이터로부터 새로운 차원이 추가 학습됨.

### PC별 해석 (DDragon stat과의 Pearson 상관 기준)

#### PC1: "교전 접근성" (Engagement Range)

| 상관 | DDragon stat |
|------|-------------|
| **+0.82** | attackrange |
| **-0.76** | attackdamage |
| **-0.70** | armor |
| **-0.60** | movespeed |
| **+0.55** | magic |
| **-0.56** | attack |

- PC1 (+) 방향: **원거리 마법 챔피언** — Seraphine, Yuumi, Vel'Koz, Nami (Support/Mage)
- PC1 (-) 방향: **근접 물리 챔피언** — K'Sante, Darius, Sion, Kayn (Tank/Fighter)
- **해석**: PC1은 "근접 교전형 vs 원거리 지원형" 축을 포착. 전체 분산의 37~46%를 설명하므로,
  LoL 드래프트에서 **가장 중요한 단일 구분 축은 "교전 접근 방식"**이다.

#### PC2: "어쌔신-다이버 vs 원딜-탱크" (Solo Agency)

| 상관 | 특징 |
|------|------|
| 약한 상관 | DDragon stat과 직접 매핑 어려움 |

- PC2 (+) 방향: Akali, Fizz, Qiyana, Diana, Vi — **솔로킬 특화 다이버/어쌔신**
- PC2 (-) 방향: Malphite, Kai'Sa, Jhin, Draven — **팀 의존형 원딜 + 이니시에이터**
- **해석**: PC2는 DDragon 수치와의 상관은 낮지만, 게임 내 행동 양식을 반영.
  **"개인 플레이 주도권(solo agency)"** 축으로 해석 가능.
  높은 PC2 = 혼자서 게임을 결정짓는 챔피언, 낮은 PC2 = 팀 협동이 필수인 챔피언.

#### PC3: "자가 생존형 원딜 vs 순수 탱크" (Self-Sufficiency)

- PC3 (+) 방향: Vayne, Yunara, Twitch, Quinn, Tristana — **자체 기동력/생존기 보유 원딜**
- PC3 (-) 방향: Maokai, Amumu, Tahm Kench, Cho'Gath — **순수 탱크 (CC·보호 특화)**
- **해석**: PC3은 Marksman 내에서 자가 생존(kiting) 능력이 높은 원딜과,
  Tank 내에서 CC에 특화된 순수 탱크를 구분. **"자립도(self-sufficiency)"** 축.

### PCA 요약

임베딩 공간의 상위 3개 축이 포착하는 것:
1. **교전 접근성** (원거리/마법 vs 근접/물리) — 46%
2. **솔로 에이전시** (암살/다이브 vs 팀 의존) — 18%
3. **자립도** (자가생존 원딜 vs 순수 탱크) — 13%

이 세 축만으로 전체 분산의 ~77%를 설명. DDragon 수치 자체로는 설명되지 않는
게임 내 행동 양식(solo agency, self-sufficiency)이 PC2-3에서 드러남.

---

## 모델 간 비교 종합

| 관점 | Model A (프로) | Model B (솔로 전체) | Model C (솔로 고티어) |
|------|--------------|-------------------|---------------------|
| 학습 정도 | 약함 (epoch 2, 10K) | 강함 (epoch 9, 137K) | 약함 (epoch 5, 2.5K) |
| Δweight 크기 | ~0.08 | ~1.3 (15×) | ~0.08 |
| 임베딩 변화 | DDragon 구조 유지 | 실질 재학습 | DDragon 구조 유지 |
| Archetype 분리 | 명확 (정형 드래프트) | 뭉개짐 (비정형) | 중간 |
| PCA 분산 집중 | PC1에 46% | PC1에 38% (분산) | PC1에 46% |

### 핵심 발견

1. **데이터량이 임베딩 품질의 결정적 요인**: B만이 DDragon 초기화를 탈피하여 독립적 임베딩 구조를 학습함.
   A/C는 데이터 부족으로 초기화 편향에 갇힘.
2. **프로와 솔로랭크의 드래프트 구조가 근본적으로 다름**: A에서 명확한 조합 원형이 보이는 반면,
   B에서는 개인 선호에 의한 노이즈가 지배적.
3. **Azir, K'Sante, Skarner가 25.19+ 패치에서 가장 큰 정체성 변동**: DDragon 설계 vs 실제 운용의 괴리가 가장 큼.
4. **PCA가 드러낸 잠재 축**: 교전 접근성 → 솔로 에이전시 → 자립도 순으로, 드래프트에서 챔피언의
   기능적 역할을 결정짓는 계층 구조가 존재.

---

## 한계 및 향후 과제

- Model A/C의 학습 부족: 데이터 증강(augmentation) 또는 더 긴 학습이 필요
- Blue side 승률 편향(53.3%)을 모델이 학습했을 가능성 — side 정보를 입력에서 분리하는 실험 필요
- PCA의 주성분 해석은 상관 기반 추론이므로 인과관계가 아닌 연관성
- Co-occurrence 분석을 Red팀과 교차하여 "카운터 관계"까지 확장 가능
