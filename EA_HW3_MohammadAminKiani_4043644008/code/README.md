# Multi-Objective Recommender Systems with Evolutionary/Swarm Optimization (HSA & MA)

This repository contains a **Google Colab-ready** project for building and evaluating a **multi-objective recommender system** on the **MovieLens 100K** dataset.  
We start from a **Matrix Factorization (MF)** baseline, then generate recommendations using two metaheuristics:

- **HSA** — Harmony Search Algorithm (swarm-inspired / memory-based search)
- **MA** — Memetic Algorithm (Genetic Algorithm + Local Search)

Both algorithms are tested under **two multi-objective strategies**:

1. **Pareto dominance (multi-objective / NSGA-II style selection)**
2. **Weighted-sum (single objective with penalty)**

The final comparison includes **5 models**:

- MF baseline
- HSA–Pareto
- HSA–Weighted
- MA–Pareto
- MA–Weighted

We evaluate them in terms of **accuracy, diversity, constraint satisfaction, catalog coverage, and exposure fairness**.

---

## 📌 Problem Statement

Given a user and a set of candidate items (top-N by MF predicted score), we must select **K items** that:

- maximize **Accuracy** (MF predicted ratings)
- maximize **Diversity** (intra-list diversity based on genre Jaccard distance)
- satisfy a **Category/Genre constraint**:
  - each recommendation list must contain at least **MIN_DISTINCT_GENRES** distinct genres

This creates a classic **trade-off** problem: improving diversity often reduces accuracy.

---

## ✅ Key Features

- **Colab Notebook** with complete runnable pipeline (data loading → MF candidates → EA optimization → evaluation → plots)
- Two multi-objective approaches (**Pareto vs Weighted**) implemented and compared
- Two algorithms (**HSA and MA**) implemented for the multi-objective selection step
- Outputs:
  - recommendation files for each model (`.csv`)
  - **trade-off plots**
  - **coverage curves**
  - **exposure distribution + Lorenz curve**
  - **genre exposure plots**
  - summary evaluation table (5-row comparison)

---

## 📂 Repository Structure

```text
.
├── notebooks/
│   ├── EA_exercise_script.ipynb
│   └── EA_exercise_script_fixed_HSA_MA.ipynb
├── data/
│   ├── raw/               # MovieLens raw files (u.data, u.item, u.user, u.genre)
│   └── processed/         # cleaned/intermediate outputs created by notebook
├── results/
│   ├── results_mf/        # MF candidate recommendation files
│   └── results_ea/        # EA results (HSA/MA × Pareto/Weighted)
├── src/                   # optional (if you modularize)
│   ├── ea_algorithms.py
│   └── metrics.py
├── README.md
└── LICENSE
```

> If you keep everything inside the notebook, `src/` can be omitted.

---

## 🚀 Quick Start (Google Colab)

### 1) Open the notebook

Open:

- `notebooks/EA_exercise_script_fixed_HSA_MA.ipynb`

### 2) Mount Google Drive

The notebook writes results to Drive paths like:

```python
/content/drive/MyDrive/Movielens_DataSet/results_ea
```

Make sure your Drive folders match or update the paths in the notebook.

### 3) Provide MovieLens data

Place MovieLens100K files in:

```text
data/raw/
  u.data
  u.item
  u.genre
  u.user
```

Or keep them in Drive and update the paths in the notebook.

### 4) Run all cells

The notebook will:

1. Load dataset and genres
2. Build MF baseline and top-N candidate pool per user
3. Run EA optimization for 4 settings:
   - HSA–Pareto
   - HSA–Weighted
   - MA–Pareto
   - MA–Weighted
4. Save output CSV files to `results/results_ea/`
5. Evaluate all 5 methods and generate plots

---

## ⚙️ Configuration

Main parameters:

| Parameter             | Meaning                              | Typical Value |
| --------------------- | ------------------------------------ | ------------- |
| `K`                   | number of recommended items per user | 10            |
| `N_CANDIDATES`        | candidate pool size from MF          | 300           |
| `MIN_DISTINCT_GENRES` | constraint threshold                 | 3             |
| `pop_size`            | EA population size                   | 50            |
| `n_gens`              | iterations/generations               | 50            |
| `w_score`             | weighted-sum accuracy weight         | 0.7           |
| `w_div`               | weighted-sum diversity weight        | 0.3           |
| `penalty_lambda`      | constraint violation penalty         | 2.0           |

### Fair comparison rule

To ensure fairness across experiments:

- use the **same** `K`, `N_CANDIDATES`, population size, number of generations, and random seed across all methods.

---

## 🧠 Algorithms

### HSA (Harmony Search Algorithm)

A memory-based stochastic search:

- **HMCR**: probability of sampling from harmony memory
- **PAR**: probability of pitch adjustment (perturbation)

**Strengths**

- Fast and stable
- Works well with Pareto archive selection

**Weaknesses**

- Can converge early if diversity operators are weak

---

### MA (Memetic Algorithm)

Genetic algorithm + local search:

- Tournament selection
- Crossover + mutation
- Local search improves individuals

**Strengths**

- High-quality solutions if time allows
- Strong exploitation via local search

**Weaknesses**

- Much slower than HSA (local search is expensive)
- Harder to scale for many users/candidates

---

## 🎯 Multi-Objective Strategies

### Pareto Dominance (NSGA-II style selection)

- Maintains a set of **non-dominated solutions**
- Uses non-dominated sorting + crowding distance
- Best for analyzing a **Pareto frontier**

### Weighted Sum + Penalty

- Converts into a single scalar objective:

\[
w*{score}\cdot Accuracy + w*{div}\cdot Diversity - \lambda\cdot violation^2
\]

- Good if you need one fixed operational policy (one final recommendation style)

---

## 📊 Evaluation Metrics

We evaluate on held-out test items per user:

### Accuracy / Ranking

- Precision@K
- Recall@K
- NDCG@K
- MRR@K

### Diversity

- ILD (Intra-List Diversity) via genre-based Jaccard distance

### Catalog-Level & Fairness

- Coverage@K (train/all catalog)
- Exposure distribution (log-log)
- Gini coefficient (inequality)
- Normalized entropy (spread)
- Lorenz curve

### Constraint

- `Constraint_ok_rate`: fraction of users whose list meets the minimum genre constraint

---

## 📈 Plots Produced

- **Accuracy vs Diversity trade-off scatter** for all 5 methods
- **Coverage vs K** curve (MF + 4 EA variants)
- **Exposure distribution** (log-log)
- **Lorenz curve** (exposure inequality)
- **Genre exposure line plot** across all recommendations
- Example user recommendation table comparing MF vs all EA variants

---

## 🧪 Output Files

Example EA result filenames:

```text
results/results_ea/
  ea_HSA_pareto_top10_from_top300_candidates.csv
  ea_HSA_weighted_top10_from_top300_candidates.csv
  ea_MA_pareto_top10_from_top300_candidates.csv
  ea_MA_weighted_top10_from_top300_candidates.csv
```

MF baseline:

```text
results/results_mf/
  mf_top10_from_top300_candidates.csv
```

Each file format:

| column      | meaning                |
| ----------- | ---------------------- |
| user_id     | user identifier        |
| item_id     | recommended movie id   |
| rank        | position in top-K list |
| pred_rating | MF predicted score     |

---

## 🛠️ Performance Notes (Runtime)

- **HSA** is typically much faster than MA.
- **MA** can become slow due to local search repeated per user.
- GPU usually does **not** significantly help unless you vectorize heavy parts (this pipeline is mostly CPU + Python loops).

If MA is too slow:

- reduce `pop_size`, `n_gens`, or `LS_PROB` **only if allowed** by your course rules
- or run fewer users for debugging, then full run for final results

---

## ✅ Reproducibility Checklist

- [ ] Fixed random seed (`seed = 42`)
- [ ] Same candidate pool from MF (`N_CANDIDATES`)
- [ ] Same EA settings for all variants (population, iterations)
- [ ] Same evaluation split and test protocol

---

## 📌 Citation

MovieLens Dataset:

- GroupLens Research, MovieLens 100K

If you use this repository in academic work, cite both MovieLens and the relevant metaheuristic references (Harmony Search, Memetic Algorithms, NSGA-II concepts).

---

## 📜 License

ui.ac.ir

---

## 👤 Author

- **Amin Kiani**
