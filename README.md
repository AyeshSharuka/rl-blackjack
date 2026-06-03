# Blackjack RL Agent - Q-Learning vs Monte Carlo

> MSc Artificial Intelligence - Reinforcement Learning (B9AI105 · CA1) | Dublin Business School

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-2.0-blue?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-orange?style=flat)
![RL](https://img.shields.io/badge/RL-Q--Learning%20%7C%20Monte%20Carlo-green?style=flat)
![Colab](https://img.shields.io/badge/Environment-Google%20Colab-orange?style=flat&logo=googlecolab&logoColor=white)

---

## Project Overview

Two Reinforcement Learning agents trained to play Blackjack - **Q-Learning** and **Monte Carlo** - built entirely from scratch without pre-built RL libraries. The project compares both algorithms across training performance, convergence speed, learned policies, and player edge over 200,000 episodes each.

Built as part of the **Reinforcement Learning (B9AI105 · CA1)** module at Dublin Business School.

---

## Problem Formulation

Blackjack is an ideal RL testbed because it has:
- A clear **state space** (player total, dealer upcard, soft hand flag, doubled flag)
- A discrete **action space** (Hit, Stand, Double Down)
- A well-defined **reward signal** (+1 win, −1 loss, 0 draw, ±2 for double down)
- **Stochastic transitions** (card draws) that make simple rule-based strategies suboptimal

RL is justified here because the optimal policy cannot be trivially derived — it requires learning through thousands of interactions with the environment.

---

## Environment Design

The Blackjack environment is implemented **fully from scratch**:

| Component | Implementation |
|---|---|
| Deck | 8-deck shoe, auto-reshuffles when < 20 cards remain |
| State | `(player_total, dealer_upcard, is_soft_hand, has_doubled)` |
| Actions | `H` = Hit · `S` = Stand · `D` = Double Down (first 2 cards only) |
| Reward | `+1` win · `-1` loss · `0` draw · `±2` for double down |
| Dealer rule | Dealer hits until 17+ (standard casino rules) |

---

## Algorithms

### Q-Learning (Temporal Difference)
Updates the Q-table **after every single step** using the Bellman equation:

```
Q(s,a) ← Q(s,a) + α [r + γ · max Q(s',a') − Q(s,a)]
```

- Learning rate `α = 0.01`, Discount factor `γ = 0.95`
- Epsilon-greedy exploration: ε decays from `1.0 → 0.05` over training

### Monte Carlo (First-Visit)
Waits until the **end of each full episode**, then updates every visited state using the actual return G:

```
G ← γG + r  (computed backwards)
Q(s,a) ← Q(s,a) + α [G − Q(s,a)]
```

- Learning rate `α = 0.02`, Discount factor `γ = 1.0`
- Same epsilon-greedy decay as Q-Learning for fair comparison

---

## Experiments & Results

### Training
- **200,000 episodes** per agent
- Reward logged every 1,000 episodes → `experiment_logs/`

### Hyperparameter Experiment
Q-Learning trained with 3 different learning rates (`α = 0.001, 0.01, 0.1`) over 100,000 episodes to analyse sensitivity.

### Convergence Analysis
Rolling window method (window = 10,000 episodes, δ < 0.005) to detect when each agent stabilised.

### Evaluation
Both agents evaluated for **50,000 games** using a **greedy policy** (no exploration):

| Metric | Q-Learning | Monte Carlo |
|---|---|---|
| Win % | *See logs* | *See logs* |
| Loss % | *See logs* | *See logs* |
| Draw % | *See logs* | *See logs* |
| Player Edge | *See logs* | *See logs* |

> Full results available in `experiment_logs/evaluation_results.json`

---

## Visualisations (9 Charts)

| Chart | Description |
|---|---|
| Learning Curves | Rolling average reward over 200k episodes for both agents |
| Win/Loss/Draw | Side-by-side bar comparison at evaluation |
| Player Edge | Net advantage (Win% − Loss%) for each agent |
| Convergence | Window-average reward showing stabilisation point |
| Hyperparameter Experiment | Learning rate α comparison (0.001 vs 0.01 vs 0.1) |
| Q-Table Size | Number of unique states explored by each agent |
| Q-Learning Policy Heatmap | Learned action per (Player Total × Dealer Upcard) |
| Monte Carlo Policy Heatmap | Learned action per (Player Total × Dealer Upcard) |
| Reward Distribution | Histogram of rewards in last 50k episodes |

---

## How to Run

### Option 1 - Google Colab (Recommended)
1. Open `rl_blackjack.py` in Colab (or the `.ipynb` version)
2. Click **Runtime → Run all**
3. Training takes ~5–10 minutes; `experiment_logs.zip` downloads automatically

### Option 2 - Local
```bash
# Clone the repo
git clone https://github.com/AyeshSharuka/rl-blackjack.git
cd rl-blackjack

# Install dependencies
pip install -r requirements.txt

# Run
python rl_blackjack.py
```

Results and charts will be saved to `experiment_logs/`.

---

## Key Hyperparameters

| Parameter | Q-Learning | Monte Carlo |
|---|---|---|
| Episodes | 200,000 | 200,000 |
| Learning rate (α) | 0.01 | 0.02 |
| Discount factor (γ) | 0.95 | 1.0 |
| Initial epsilon | 1.0 | 1.0 |
| Min epsilon | 0.05 | 0.05 |
| Epsilon decay | 0.99999/episode | 0.99999/episode |
| Decks in shoe | 8 | 8 |

---

## Dependencies

```
numpy>=2.0
matplotlib>=3.10
seaborn>=0.13
pandas>=2.2
```

No RL libraries used - environment and agents are built entirely from scratch.



