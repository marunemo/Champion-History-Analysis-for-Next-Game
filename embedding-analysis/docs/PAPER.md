# What Does a Win-Prediction Model Learn About Champions?
### Interpreting the Learned Champion-Embedding Space of League of Legends Drafts across Tiers and Architectures

**Author:** _(TBD)_ · **Code:** `embedding-analysis/` · **Data:** Champion-History-Analysis-for-Next-Game

---

## Abstract

We train models that predict the winner of a League of Legends match from its draft
(the ten chosen champions) and then, rather than chasing accuracy, **interpret the champion
embedding space those models learn**. Two architectures — a position-agnostic feed-forward
network (FFNN) and a 1-D convolutional network (CNN) — are trained on four datasets
(professional esports, all-tier solo queue, Challenger, and Iron–Silver), each initialised
from a 32-dimensional projection of DDragon champion metadata. As expected for a draft-only
task, validation accuracy is capped at 51–53%; the contribution is not the classifier but a
four-way analysis (t-SNE, KMeans archetypes, Δweight, PCA) of the resulting embeddings. We
find that **FFNN ≈ CNN** (pick-slot order carries no win signal), that embedding
reorganisation is driven by **data volume *and* draft diversity** (only the 137K all-tier
set escapes its initialisation), and that the dominant latent axis (PC1, 37–46% variance)
encodes **engagement range** — correlating r = +0.82 with `attackrange`, a feature that was
**never an input** — evidence that the model picked up an unstated mechanic indirectly. A
second axis (PC2, 16–18%)
captures a "solo agency" dimension absent from DDragon stats. High- and low-tier drafts
diverge sharply (assassin-divers vs. utility champions), the largest meta deviations are
Azir, K'Sante and Skarner, and a transfer-learning study shows the benefit of a pretrained
source depends on **domain similarity**, not merely on target scarcity.

---

## 1. Introduction

In League of Legends two teams of five draft champions before play begins. We ask how much
of the outcome is recoverable from the draft alone, and — more importantly — **what a model
trained on that signal comes to believe about champions**. The win-prediction task is treated
as a *probe*: a 32-dimensional champion embedding, seeded from designer-defined metadata and
then fine-tuned on real match results, is a learned representation whose geometry we can read
as game knowledge.

The goal is explicitly **not predictive accuracy** — draft-only prediction is structurally
low-signal (prior work caps it near 55%, the lineup-only ceiling familiar from sports
analytics), since outcomes are dominated by player skill and in-game execution. The goal is
to ask: when the embedding moves away from its DDragon initialisation, *which* champions move,
*along which axes*, and *how does this differ across competitive environments and across
architectures*.

## 2. Data

Four label-encoded datasets share one encoder (**192 champions**, alphabetically ordered,
after dropping DDragon-unsupported champions and applying 21 display→key name fixes):

| Dataset | Rows | Environment | Filter |
|---|---|---|---|
| A `pro` | 10,008 | Professional esports (Oracle's Elixir 2025) | position = team |
| B `solo_all` | 137,146 | Solo queue, all tiers | QueueType = CLASSIC |
| C `solo_high` | 2,566 | Solo queue, GM + Challenger | RankName ∈ {Grandmaster, Challenger} |
| D `solo_low` | 13,984 | Solo queue, Iron–Silver | rank ∈ {Iron, Bronze, Silver} |

Each row is `blue_p1..5`, `red_p1..5` (encoded champion ids) and `result` (1 = blue win).

**Champion feature matrix (17-d).** The embedding initialisation is built from DDragon 15.19.1:
`feat = [tags one-hot (6) | info ÷10 (4) | stats min-max (7)]`, i.e. six role tags
(Fighter/Tank/Mage/Assassin/Marksman/Support), four info ratings (attack, defense, magic,
difficulty) and seven base stats (hp, armor, spellblock, attackdamage, attackspeed, movespeed,
hpregen). **`attackrange` is deliberately excluded** — a fact that becomes central in §4.3.

**Embedding initialisation (`embedding_init.pt`).** A random Xavier-uniform linear map
`Linear(17 → 32)` is applied once under `no_grad` to the feature matrix, giving a `(192, 32)`
starting point: champions with similar DDragon attributes start nearby, but the 32 axes carry
no intrinsic meaning ("better than random", not a semantic space). The embedding is then
trained with `requires_grad = True`.

## 3. Methods

### 3.1 Architectures
- **FFNN (`DraftEmbeddingFFNN`, ~105K params).** Embed all ten picks, **concatenate**
  (blue 5 ∥ red 5 → 320), then `Linear(320,256)→ReLU→Dropout(0.3)→Linear(256,64)→ReLU→
  Linear(64,1)→Sigmoid`. Pick-slot order is ignored.
- **CNN (`DraftEmbeddingCNN`, ~23K params).** Stack the ten picks as a `(10, 32)` sequence,
  `permute` to `(32, 10)`, then `Conv1d(32,64,k=3)→ReLU→Conv1d(64,128,k=3)→ReLU→
  AdaptiveAvgPool1d(1)→Linear(128,64)→ReLU→Dropout(0.3)→Linear(64,1)→Sigmoid`. Treats the
  draft as a 1-D sequence, able to capture local adjacency interactions.

### 3.2 Training & swap augmentation
Adam (lr 1e-3), BCELoss, ≤30 epochs, early stopping (patience 5) on val loss, batch 256,
80/20 train/val split (seed 42). **Blue/Red swap augmentation** doubles the *training* data
only: every `(blue, red, result=1)` is mirrored to `(red, blue, result=0)`, removing the
blue-side win-rate bias (53.3% in pro) so the model cannot learn "blue ⇒ win". Validation is
left un-augmented to avoid leakage.

### 3.3 Embedding analyses
1. **t-SNE (before/after)** of the `(192, 32)` embedding, coloured by the six DDragon role
   tags, plus a *migration* overlay arrowing the before→after move of the top-Δ champions.
2. **Archetype clustering** — mean-pool each blue 5-pick to `(N, 32)`, `KMeans(k=7)`,
   visualised with t-SNE, to quantify how *formulaic* team compositions are.
3. **Δweight** — `‖W_after − W_before‖₂` per champion, ranking deviation from the DDragon
   identity.
4. **PCA** — principal axes of the learned embedding, with Pearson correlation of each axis
   against DDragon stats, including the held-out `attackrange`.

### 3.4 Transfer learning (B → A/C/D)
Because A/C/D are data-poor and never leave their initialisation, we re-initialise them from
**Model B's learned embedding** (`--init-weights embedding_B_learned.pt`) instead of the
DDragon projection, keeping all other hyper-parameters fixed, to test whether all-tier
solo-queue knowledge transfers.

## 4. Results

### 4.1 Training (validation)

| Model | Rows | FFNN epoch / acc | CNN epoch / acc |
|---|---|---|---|
| A pro | 10,008 | 1 / 53.1% | 7 / 52.8% |
| B solo_all | 137,146 | 7 / **53.2%** | 7 / 53.1% |
| C solo_high | 2,566 | 4 / 52.7% | 10 / 51.4% |
| D solo_low | 13,984 | 2 / 51.8% | 8 / 51.3% |

FFNN and CNN differ by < 0.003 val loss everywhere; B reaches the lowest val loss (0.6897).
CNN consistently needs more epochs (Conv1d deforms the embedding more slowly). Swap
augmentation stabilised training and lifted the smallest set, **Model C, from 49.6% → 52.7%**.

### 4.2 Embedding reorganisation — data volume *and* diversity
Across t-SNE, Δweight and PCA, **only Model B (137K, all tiers) escapes its DDragon
initialisation**; A, C and D retain the initial structure. A is data-limited; C is severely
data-limited (2.5K); D has *more* data than A yet still barely moves — low-tier drafts
concentrate on a few popular champions and lack the *combinatorial diversity* needed to
reorganise the whole space. Embedding learning therefore needs both volume and draft diversity,
which only B satisfies.

### 4.3 PCA latent axes

| PC | Variance | Interpretation | Top DDragon correlation |
|---|---|---|---|
| 1 | 37–46% | **Engagement range** (ranged support ↔ melee engage) | attackrange **+0.82**, attackdamage −0.77, armor −0.67 |
| 2 | 16–18% | **Solo agency** (solo-kill/roam ↔ team-dependent) | weak — a purely data-learned axis |
| 3 | 13–15% | **Frontline identity** (tank ↔ self-sufficient carry) | defense +0.57, attack −0.41 |

PC1 extremes: (+) Milio, Vel'Koz, Sona, Seraphine; (−) K'Sante, Darius, Nocturne, Jarvan IV.
PC2 extremes: (+) Nilah, Fizz, Akali, LeBlanc; (−) Ezreal, Corki, Varus, Yunara.

The headline result: **`attackrange` was not an input feature**, yet PC1 correlates with it at
r = +0.82 — the model recovered an unstated game mechanic (range) from draft win/loss patterns
(partly via tag one-hots encoding range indirectly, partly via ranged- vs. melee-composition
win differences). PC2 has no strong DDragon correlate; it is a dimension visible only in usage
patterns. Per-model variance confirms §4.2: A/C/D sit near 46/18/13%, while **B alone shrinks
PC1 (37.7%) and grows PC2–PC3 (15.6/15.1%)** — given enough data, the DDragon-dominated axis
recedes and game-derived axes emerge.

### 4.4 Archetypes and the tier divide
Composition becomes *less* formulaic from pro → high → all ≈ low tier. Pro (A) shows clean
archetypes ("tank frontline", "AP-top engage", "ranged poke") because teams draft designed
compositions; solo queue (B) blurs them because five players pick independently. The most
interpretive contrast is **C vs. D at similar size**: Challenger (C) is dominated by
assassin-divers (Ambessa, Zed, Akali, Qiyana) — *individual skill expression* — while
Iron–Silver (D) converges on utility champions (Lux, Miss Fortune, Teemo, Malphite) —
*executional stability*. The definition of a "good draft" itself is tier-dependent.

### 4.5 Δweight — deviation from designer identity

| Model | Δ magnitude | Top-3 deviating champions |
|---|---|---|
| A pro | ~0.08 | Morgana, Jarvan IV, Poppy (pro role swaps) |
| B solo_all | **~1.3** | **Azir (1.75), K'Sante (1.60), Skarner (1.57)** |
| C solo_high | ~0.08 | Ashe, Zeri, Sett |
| D solo_low | ~0.08 | (stays at DDragon init) |

B's top deviators all underwent identity shifts around patch 25.19: Azir (tagged Mage, played
as a ranged-DPS/peel hybrid), K'Sante (tagged Tank, played as a 1v1 top bruiser), Skarner
(reworked jungle-tank → initiator). The model learned the gap between *designed* and *played*
roles purely from win/loss data — i.e. the embedding captures game function, not just statistics.

### 4.6 Transfer learning depends on domain distance

| Target | DDragon-init FFNN | B-init FFNN | Δ |
|---|---|---|---|
| A pro | 53.1% | 52.5% | −0.6 |
| C solo_high | 52.7% | 49.2% | **−3.5** |
| D solo_low | 51.8% | **54.6%** | **+2.8** |

Only **D improves**: B's general solo-queue meta is a useful prior for the low-tier tail of
rarely-drafted champions (D_tl also has the lowest val loss overall, 0.6892). **C degrades** —
B's "all-tier average" dilutes Challenger's extreme assassin-diver preference (source too far
from target). **A is neutral** — pro and solo queue are qualitatively different environments.
This contradicts the assumption that scarce data always benefits from transfer: source–target
domain distance is the deciding variable.

### 4.7 Figures
- `outputs/figures/tsne/*.png` (+ `*_migration.png`) — before/after role clusters and the
  Azir/K'Sante/Skarner migration arrows.
- `outputs/figures/archetype/*.png` — KMeans composition archetypes per model.
- `outputs/figures/delta_shift/*.png` — per-champion Δweight rankings.
- `outputs/figures/pca/*.png` (+ `*_annotated.png`) — scree, PC1–PC2 / PC1–PC3 with labelled
  extremes.

## 5. Discussion

**Identity over order.** FFNN ≈ CNN (< 0.003 val loss) means pick-slot position p1…p5 carries
no win signal: in a LoL draft *who* is present matters, not *where*. (Caveat: these slots are
likely position listings, not true pick order, so this is not a claim about draft *sequence*.)

**Data shapes the space.** Embedding quality is governed by data volume *and* draft diversity;
B is the only model that re-derives game structure, and its emergence of PC2/PC3 shows prior
knowledge (DDragon) receding as evidence accrues.

**The model discovers mechanics.** Recovering `attackrange` (r = +0.82) without it as input,
and relocating exactly the champions (Azir, K'Sante, Skarner) whose played role diverges from
their designed one, are independent signs that the embedding encodes functional game meaning.

**Cross-analysis agreement.** t-SNE, Δweight, PCA and archetypes converge on the same
conclusions, mitigating the over-interpretation risk of any single visualisation.

## 6. Limitations & Future Work

- The initialisation set `torch.manual_seed`, so `embedding_init.pt` is not reproducible
  on regeneration (the current file is pinned).
- PCA interpretation is correlational, not causal.
- Counter relationships are unanalysed; blue–red cross co-occurrence could quantify counter
  pairs.
- The meaning of p1…p5 is uncertain (pro draft order vs. solo-queue position listing), which
  may have blunted any CNN advantage.
- Transfer sources other than B (e.g. a nearer A→C pairing) are worth testing.

## 7. Conclusion

Predicting a League of Legends match from its draft alone is near a coin flip, but the
*embedding* a win-prediction model learns is legible. It separates champions by engagement
range and solo agency, recovers an unstated range mechanic, flags the champions whose meta
identity has drifted from their design, and exposes a fundamental high- vs. low-tier
difference in what a draft is *for*. The embedding is best read not as a predictor but as a
**data-grounded map of how champions actually function across competitive environments**.

---

*Reproduce with the train/analyze commands in [`../README.md`](../README.md). Reports in
`outputs/`, figures in `outputs/figures/`.*
