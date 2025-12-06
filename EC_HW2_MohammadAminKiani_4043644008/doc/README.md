# Portfolio Optimization with Genetic Algorithm (GA) & Particle Swarm Optimization (PSO)

This repository contains the complete implementation of my **Evolutionary Computation** homework on
portfolio optimization.  
The goal is to build a small stock portfolio that **maximizes the Sharpe ratio** using two metaheuristic
algorithms:

- a **Genetic Algorithm (GA)** (evolutionary algorithm)
- a **Particle Swarm Optimization (PSO)** (swarm-based metaheuristic)

The README is written so that **someone with no background in finance or optimization** can still follow
what is happening and reproduce the results.

---

## 1. Intuition: What problem are we solving?

Imagine you have some money and you want to invest it in the stock market. You can split your money across
different stocks (assets). The question is:

> **How much of my money should I put in each stock so that I get the best “return per unit of risk”?**

This is exactly what **portfolio optimization** does.  
Here:

- A **portfolio** = a combination of stocks + a weight (percentage of money) for each one.
- **Risk** = how much the returns fluctuate (volatility).
- **Return** = how much, on average, your investment grows.

To combine return and risk into a single metric, we use the famous **Sharpe ratio**.

---

## 2. The Sharpe Ratio (objective function)

The **Sharpe ratio** measures “return per unit of risk”.  
Mathematically:

\[
\text{Sharpe} = \frac{\mathbb{E}[R_p - R_f]}{\sigma(R_p - R_f)}
\]

where:

- \( R_p \) = portfolio return in a given period (here: one week),
- \( R_f \) = risk-free return in the same period (e.g. a bank deposit),
- \( R_p - R_f \) = **excess return** of the portfolio,
- \( \mathbb{E}[\cdot] \) = average (mean) over time,
- \( \sigma(\cdot) \) = standard deviation (volatility).

**Interpretation:**  
- Large positive Sharpe → good: high excess return *relative* to its risk.  
- Small or negative Sharpe → bad: low or negative excess return, or too much volatility.

In this project, **our fitness function is exactly the Sharpe ratio**.  
We **maximize** it.

---

## 3. Dataset

The dataset is a CSV file, e.g.:

```text
Dataset.csv
```

It contains **real weekly returns** of **50 top S&P 500 stocks** plus the weekly **risk-free rate**.

- Column **`Week`**: time index (week).
- Columns **2 to 51**: weekly returns of each stock (in percent) based on closing prices.
- Column **`Rf`**: weekly **risk-free rate** (also in percent).

Example (simplified):

```text
Week,BMY,MSFT,...,XOM,Rf
2021-06-01,0.058691,0.0123,...,0.0345,0.001
...
```

> If a stock has value `0.058691` in a week, it means its price increased by about 5.8% compared to the previous week.

---

## 4. Optimization problem (constraints & design)

We want to build a portfolio **from these 50 stocks** with the following constraints:

1. The portfolio must contain **exactly 10 stocks** with **non-zero weight**.
2. All weights must be **non-negative** (no short-selling).
3. The **sum of all weights = 1** (100% of capital is invested).
4. The objective is to **maximize the weekly Sharpe ratio** (and implicitly, also its annualised value).

This is a **combinatorial & continuous** optimization problem:

- Combinatorial: choose **which** 10 stocks out of 50 → there are `C(50, 10)` possible subsets!
- Continuous: decide **what weight** to assign to each of the chosen stocks.

Because the search space is huge and the constraints are non-trivial, we do **not** use classical
closed-form methods. Instead, we use **metaheuristics** (GA & PSO).

---

## 5. Representation of a solution (chromosome / particle)

To handle both “which stocks” and “what weights” in a simple way, we do the following:

- Each candidate solution is a **vector of length 50** with real values in `[0, 1]`.
- Let’s call this vector `chromosome` (in GA) or `position` (in PSO).

To convert a chromosome into an actual portfolio:

1. **Sort** the 50 values and select the indices of the **top 10** entries.
2. Take these 10 values and **normalise** them so that their sum = 1.
3. These 10 normalised values become the **weights** of the 10 selected stocks.
4. All other 40 weights are set to **0**.

This way:

- We **always** have exactly **10 assets** in the portfolio.
- We **avoid complicated penalty functions** for violating the “10-asset” constraint.
- The metaheuristic only needs to search in `[0,1]^{50}`, which is easy to handle.

---

## 6. Genetic Algorithm (GA)

The **Genetic Algorithm** is an evolutionary method inspired by natural selection.

### 6.1. GA steps

1. **Initial population**:  
   - Create `pop_size` random chromosomes (each is a 50-dimensional vector in `[0,1]`).
2. **Fitness evaluation**:  
   - For each chromosome:
     - Decode it into a 10-asset portfolio.
     - Compute its **Sharpe ratio**.
3. **Selection (tournament)**:  
   - Randomly pick, e.g., 3 individuals.
   - The one with the highest Sharpe wins and becomes a parent.
4. **Crossover (two-point)**:  
   - Take two parents, cut their chromosomes at two random positions, and swap the middle segments.
5. **Mutation**:  
   - With some probability, add small Gaussian noise to some genes and clip to `[0,1]`.
6. **Elitism**:  
   - Copy a small number of the best individuals unchanged to the next generation.
7. **Repeat** for a fixed number of generations (e.g., 300).

### 6.2. Why this design?

- **Tournament selection** controls selective pressure and keeps diversity.
- **Two-point crossover** recombines good substructures from different parents.
- **Gaussian mutation** supports **exploration** (searching new areas) and **exploitation** (fine-tuning weights).
- **Elitism** ensures we never lose the best found solutions.

---

## 7. Particle Swarm Optimization (PSO)

PSO is a swarm-based algorithm inspired by the behaviour of bird flocks.

### 7.1. PSO steps

- We maintain a **swarm** of particles. Each particle has:
  - a **position** (50-dimensional vector in `[0,1]`),
  - a **velocity** (same dimension),
  - a **personal best position** (pbest),
  - we also track a **global best position** (gbest) across the swarm.

At each iteration:

1. Evaluate the **Sharpe ratio** of each particle’s position.
2. Update `pbest` if the particle finds a better position.
3. Update `gbest` if any particle finds a better position than the current global best.
4. Update velocity and position via:

   \[
   v \leftarrow w\,v + c_1 r_1 (pbest - x) + c_2 r_2 (gbest - x)
   \]

   \[
   x \leftarrow x + v
   \]

   where `w` is inertia, `c1`, `c2` are cognitive/social coefficients, and `r1`, `r2` are random vectors.

5. **Clip** positions to `[0,1]` and **limit** velocities to avoid overly big jumps.

The same **decode** function (top-10 normalisation) is used to get the portfolio and compute Sharpe.

---

## 8. Project structure

A recommended structure (as used in this assignment):

```text
.
├── code/
│   ├── portfolio_ga_pso.ipynb      # Colab-ready notebook with full implementation
│   ├── ga_pso_portfolio.py         # (Optional) pure Python script version
│   └── utils.py                    # (Optional) helper functions
├── doc/
│   └── YourName_StudentID.pdf      # formal report (required by the course)
├── Dataset.csv                     # main training dataset
├── Dataset_test.csv                # (optional) hidden/extra test dataset
└── README.md                       # this file
```

> You can adapt filenames to your own setup; just update the README accordingly.

---

## 9. How to run the code

### 9.1. Option A – Google Colab (recommended)

1. Open the notebook in `code/portfolio_ga_pso.ipynb` in Google Colab.
2. Upload `Dataset.csv` to the Colab environment (same folder as the notebook).
3. Run the cells in order:
   - **Data loading & preprocessing**
   - **Sharpe ratio functions**
   - **GA implementation & runs**
   - **PSO implementation & runs**
   - **Plots (convergence & pie charts)**
   - **Train/Test evaluation** (if you also upload `Dataset_test.csv`)
4. At the end, the notebook prints:
   - best GA portfolio (stocks + weights + Sharpe),
   - best PSO portfolio,
   - Sharpe on train and test for each algorithm.

### 9.2. Option B – Local Python environment

1. Install Python 3.9+ and create a virtual environment (recommended).
2. Install dependencies:

   ```bash
   pip install numpy pandas matplotlib
   ```

3. Place `Dataset.csv` in the repository root.
4. Run the Python script (if you created one):

   ```bash
   python code/ga_pso_portfolio.py
   ```

5. The script will:
   - run GA and/or PSO,
   - print summary statistics,
   - generate plots (saved as PNG in a folder like `results/`).

---

## 10. Experiments and results (example setup)

In the report and notebook, we typically use:

- **GA parameters** (example):
  - population size: 100
  - generations: 300
  - crossover rate: 0.8
  - mutation rate: 0.1
  - mutation sigma: 0.1
  - tournament size: 3
  - elite count: 2

- **PSO parameters** (example):
  - swarm size: 60
  - iterations: 300
  - inertia weight `w`: 0.7
  - cognitive coefficient `c1`: 1.5
  - social coefficient `c2`: 1.5
  - max velocity magnitude: 0.2

We run **each algorithm 5 times independently** (different random seeds) to see the effect of randomness.

Example summary (from one of my runs, on the training dataset):

```text
GA:
  Runs: 5
  Best Sharpe (weekly):  0.2255
  Average best Sharpe:   0.2220
  Std of best Sharpe:    0.0029

PSO:
  Runs: 5
  Best Sharpe (weekly):  0.2160
  Average best Sharpe:   0.2086
  Std of best Sharpe:    0.0073
```

To interpret weekly Sharpe, we can annualise it (roughly):

\[
\text{Sharpe}_{annual} \approx \text{Sharpe}_{weekly} \times \sqrt{52}
\]

For GA with ~0.225 weekly Sharpe, this is roughly **1.6 annual Sharpe**, which is considered a **strong performance**
for a portfolio optimisation exercise.

Typically, GA produced:

- slightly **higher Sharpe** than PSO,
- and **more stable** results (lower standard deviation across runs).

PSO still found very similar portfolios (9 out of 10 selected stocks were identical to the GA solution), which shows
that both algorithms are converging to a similar optimum, but GA is slightly more robust in this setup.

---

## 11. Evaluation on unseen test data

To check whether the portfolio is **overfitting** the given dataset, we also support evaluation on **test data** with
the same format:

- `Dataset_test.csv` – provided separately (e.g. by TA).

We reuse the same evaluation function:

```python
def sharpe_ratio_on_dataframe(weights, df_any, asset_columns, rf_column="Rf"):
    rets = df_any[asset_columns].values
    rf   = df_any[rf_column].values.reshape(-1, 1)

    port_ret = rets @ weights
    excess   = port_ret.reshape(-1, 1) - rf
    excess   = excess.ravel()

    mean_excess = excess.mean()
    std_excess  = excess.std(ddof=1)
    if std_excess == 0:
        return -1e9
    return mean_excess / std_excess
```

This allows us to report, for each method (GA and PSO):

- **Sharpe on training data**,
- **Sharpe on test data** (unseen).

If the Sharpe on test is still reasonably high (not collapsing to 0 or negative), it indicates that the
portfolio generalises reasonably well and is not just memorising the training set.

---

## 12. What you can change or extend

Possible extensions and experiments:

- Try different **population/swarm sizes** and see how convergence changes.
- Modify **crossover** or **mutation** strategies (e.g. uniform crossover, non-uniform mutation).
- Add more **constraints**, such as:
  - maximum weight per stock,
  - sector constraints,
  - minimum/maximum number of stocks from a sector.
- Use alternative **risk/return measures** (e.g. downside risk, CVaR).
- Compare with a **baseline portfolio**:
  - equal-weight portfolio over all 50 stocks,
  - equal-weight over 10 randomly chosen stocks.

These extensions fit naturally into the current code by adjusting the fitness function and/or
the repair/decoding logic.

---

## 13. Requirements

- Python 3.9+
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`

(If you use the notebook in Colab, these are already available.)

---

## 14. License

You can adjust this section based on your preference. For example, an MIT License:

```text
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
...
```

---

## 15. Contact

If you have any questions about the code, algorithms, or the assignment setup, feel free to open an issue
in the GitHub repository or contact me directly.
