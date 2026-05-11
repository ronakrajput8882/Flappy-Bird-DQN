<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=200&section=header&text=🐦%20Flappy%20Bird%20DQN&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Teaching%20an%20Agent%20to%20Fly%20with%20Deep%20Q-Learning&descAlignY=60&descAlign=50" width="100%"/>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29-black?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
</div>

---

## 📌 Project Overview

A **Deep Q-Network (DQN)** agent trained from scratch to play Flappy Bird using `flappy_bird_gymnasium`. The agent learns entirely through trial and error — starting with negative rewards and progressively improving to score **18+ points** over thousands of training episodes.

Key features:
- **Experience Replay** with a 100K-capacity replay buffer for stable learning
- **Target Network** synced every 1000 steps to prevent divergence
- **Epsilon-Greedy Exploration** with exponential decay (1.0 → 0.05)
- **Dual-mode**: train from scratch or run a saved model in render mode

---

## 📂 Environment

| Parameter | Value |
|:---|:---|
| Environment | `FlappyBird-v0` (flappy_bird_gymnasium) |
| Observation Space | 12-dimensional state vector |
| Action Space | 2 (flap / no-flap) |
| Reward Signal | +1.2 per pipe passed, -1 on death, small negative per frame |

---

## 🔄 Pipeline Workflow

```
Env Reset → State Observation → ε-Greedy Action Selection → Step Env → Store in Replay Buffer → Mini-batch Sample → Bellman Update → Target Net Sync
```

1️⃣ **Environment Setup** — Gymnasium `FlappyBird-v0` with 12 state features  
2️⃣ **Policy DQN** — 2-layer MLP (12 → 256 → 2) with ReLU activation  
3️⃣ **Target DQN** — Hard-synced copy of policy network every 1000 steps  
4️⃣ **Experience Replay** — `deque`-based buffer, uniform random sampling  
5️⃣ **Epsilon Decay** — Starts at 1.0, decays by ×0.9995 per episode, floors at 0.05  
6️⃣ **Bellman Optimization** — MSELoss on current Q vs target Q, Adam optimizer (lr=0.001)  
7️⃣ **Model Persistence** — Best model checkpoint saved to `runs/flappybirdv0.pt`

---

## 🤖 Model Architecture

### 1️⃣ Deep Q-Network (DQN) ⭐ Best Model

```python
DQN(
  Linear(12, 256),
  ReLU(),
  Linear(256, 2)
)
```

- Lightweight 2-layer MLP — sufficient for low-dimensional state space
- Outputs Q-values for both actions; argmax gives the greedy action
- Separate policy and target networks prevent the moving-target instability problem
- Hard target sync (every 1000 steps) balances stability vs. responsiveness

---

## 📊 Training Progress

| Training Phase | Episodes | Best Reward |
|:---|:---:|:---:|
| Initial (random) | 1 | -8.7 |
| First positive reward | ~1432 | +0.9 |
| First pipe cleared | ~1765 | +2.1 |
| Stable flight | ~5420 | +6.9 |
| Best checkpoint | ~20950 | **+18.0** |

**Hyperparameters:**

| Parameter | Value |
|:---|:---|
| Gamma (γ) | 0.99 |
| Alpha (lr) | 0.001 |
| Epsilon Init | 1.0 |
| Epsilon Min | 0.05 |
| Epsilon Decay | 0.9995 |
| Replay Buffer Size | 100,000 |
| Mini-batch Size | 32 |
| Network Sync Rate | 1000 steps |
| Reward Threshold | 1000 |

---

## 🔍 Key Insights

- 📈 **Reward crossed 0** only after ~1432 episodes — the agent must learn to consistently avoid the first pipe before positive reward accumulates
- 🔄 **Target network stabilization** is critical — without it, Q-value overestimation causes training collapse in pixel-sparse environments
- ⚡ **Epsilon decay rate (0.9995)** is deliberately slow — FlappyBird requires extensive exploration early on due to sparse reward structure
- 🏆 **Best reward of +18.0** was reached at episode 20,950, showing the long-horizon nature of RL for reactive control tasks
- 📉 The log shows multiple training restarts — each run benefits from the previous model's checkpoint as a warm start

---

## 🗂️ Repository Structure

```
flappy-bird-dqn/
├── agent.py                # Main DQN agent — training & inference loop
├── dqn.py                  # Neural network architecture
├── expirence_replay.py     # Replay memory buffer
├── game_flappy_bird.py     # Manual play script (human mode)
├── cuda.py                 # Device/CUDA diagnostics
├── parameters.yaml         # Hyperparameter config
├── runs/
│   ├── flappybirdv0.pt     # Best saved model checkpoint
│   └── flappybirdv0.log    # Training reward log
└── README.md
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install torch flappy-bird-gymnasium gymnasium pygame pyyaml

# Train the agent
python agent.py flappybirdv0 --train

# Watch the trained agent play
python agent.py flappybirdv0

# Play manually
python game_flappy_bird.py

# Check CUDA availability
python cuda.py
```

---

## 🧠 Key Learnings

- DQN with a **target network + replay buffer** is the minimum viable setup for stable game-playing RL
- **Reward shaping matters** — the small per-frame negative reward forces the agent to learn efficiency, not just survival
- **Long training horizons** are normal for reactive control tasks — positive rewards don't appear until ~1400 episodes
- **Checkpoint-based saving** (best reward only) is more reliable than periodic saves for highly stochastic environments
- MPS/CUDA device detection should always be a first-class concern for training scripts

---

## 🛠️ Tech Stack

| Tool | Use |
|:---|:---|
| PyTorch | DQN model, loss computation, backprop |
| Gymnasium | FlappyBird-v0 environment interface |
| flappy_bird_gymnasium | Custom Flappy Bird env registration |
| PyGame | Human-mode rendering |
| PyYAML | Hyperparameter config loading |
| Python `collections.deque` | Replay memory buffer |
| CUDA / MPS | GPU-accelerated training |

---

<div align="center">

### Connect with me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/ronaksinh-rajput8882)
[![Instagram](https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white)](https://instagram.com/techwithronak)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ronakrajput8882)

*If you found this useful, please ⭐ the repo!*

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=2,12,24&height=100&section=footer" width="100%"/>

</div>