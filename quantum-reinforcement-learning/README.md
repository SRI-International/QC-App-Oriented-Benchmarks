# Quantum Reinforcement Learning (QRL) – Benchmark Program

---

**Note:** This repository is a work in progress (WIP), including this README, and will be updated over time.  

With the advent of quantum computers, **Quantum Machine Learning (QML)** has emerged as a key application area within the quantum computing ecosystem. Among its subfields, **Quantum Reinforcement Learning (QRL)** is rapidly gaining attention. QRL typically employs a hybrid quantum–classical loop, where a separate classical module—called the *environment*—interacts with the quantum agent.  

In this work, we provide a **Quantum Deep Q-Network (DQN)**[[1]](#references) implementation to solve the **FrozenLake** environment [[2]](#references) from the [Gymnasium](https://gymnasium.farama.org/) library.  

---

## Problem Outline

This benchmark implements a **Quantum Deep Q-Network (DQN)** agent for the [Gymnasium FrozenLake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment, following the approach of Kruse *et al.* [[3]](#references).

### FrozenLake Environment

<p align="center">
  <img align="center" width="134" height="127" alt="Frozen Lake Environment"
       src="https://github.com/user-attachments/assets/b8ed8fde-5ab9-43e7-b7ee-8e145c034817" />
</p>

*Fig. 1: FrozenLake map for a 4×4 grid.*

The agent begins at the upper-left tile and must reach the goal (green square) while avoiding holes (blue tiles) and staying within the grid. At each step, the agent can take one of four discrete actions: **up**, **down**, **left**, or **right**.  

- **Holes or out-of-bounds:** Episode ends, agent resets to the start.  
- **Goal (gift square):** Reward **+1**, episode ends.  
- **Safe tiles (purple):** Reward **0**, agent continues.  

The challenge lies in balancing exploration and exploitation to maximize the probability of reaching the goal.
### Parameterized Quantum Circuit Kernel




---

## Requirements  

The dependencies differ depending on the benchmark method you want to run:  

| Method | Description | Additional Requirements |
|:------:|-------------|:------------------------:|
| **1** | Benchmark QRL Ansatz | None |
| **2** | QRL Training Loop (FrozenLake) | `gymnasium` (v1.x.x) |

- If you only plan to run **Method 1**, no additional libraries need to be installed.  
- To run **Method 2**, install the Gymnasium library:  

```bash
pip install gymnasium
```
---
## Inputs

---

## References
[1] Skolik, Andrea, Sofiene Jerbi, and Vedran Dunjko. "Quantum agents in the gym: a variational quantum algorithm for deep q-learning." Quantum 6 (2022): 720.

[2] Towers, Mark, et al. "Gymnasium: A standard interface for reinforcement learning environments." arXiv preprint arXiv:2407.17032 (2024).

[3] Kruse, Georg, et al. "Benchmarking quantum reinforcement learning." arXiv preprint arXiv:2502.04909 (2025).
