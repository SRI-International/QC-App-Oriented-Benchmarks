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

We employ a **Quantum Deep Q-Network (Q-DQN)** to perform Quantum Reinforcement Learning (QRL).  
Instead of a classical deep neural network, Q-DQN uses a **parameterized quantum circuit (PQC) ansatz** as defined in Skolik et al. [[1]](#references).  

- **Qubit width**: determined by the larger of the environment’s observation space or action space.  
- **Measured qubits**: equal to the size of the action space.  

<p align="center">
<img width="754" height="259" alt="QRL ansatz" src="https://github.com/user-attachments/assets/86a46caa-4859-40ec-84a1-67a7463c3e68" />
</p>

*Fig. 2: Parameterized quantum circuit (ansatz) used in QRL.*  

The ansatz consists of layers of parameterized $R_y$ and $R_z$ rotations, while $R_x$ gates act as input encoders (applied only if the corresponding observation bit is 1).  

The benchmark also allows users to:  
- Enable or disable **data re-upload layers**  
- Specify the **number of ansatz layers**


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

The relevant parameters for methods 1 and method 2 of the QRL Benchmark are as follows.

#### Method 1 (Ansatz Benchmarking)

| Argument | Description |
|----------|-------------|
| `--min_qubits` | Minimum number of qubits to benchmark. |
| `--max_qubits` | Maximum number of qubits to benchmark. |
| `--skip_qubits` | Step size for sweeping qubit counts. |
| `--max_circuits` | Number of circuit repetitions per qubit size. |
| `--num_layers` | Number of ansatz layers in the PQC. |
| `--init_state` | Initial state to encode as input. |
| `--n_measurements` | Number of measurement operations in the circuit. |
| `--num_shots` | Number of shots per circuit execution. |
| `--data_reupload` | Enable/disable data re-uploading in the PQC. |
| `--nonoise` | Run with a noiseless simulator (ignore backend noise). |

#### Method 2 (QRL Loop)

| Argument | Description |
|----------|-------------|
| `--num_layers` | Number of ansatz layers in the PQC. |
| `--n_measurements` | Number of measurement operations (size of action space). |
| `--num_shots` | Number of shots per circuit execution. |
| `--data_reupload` | Enable/disable data re-uploading in the PQC. |
| `--total_steps` | Maximum number of training steps in the benchmark. |
| `--learning_start` | Step index after which gradient updates begin. |
| `--params_update` | Steps between parameter updates. |
| `--target_update` | Steps between target network updates. |
| `--batch_size` | Replay buffer batch size. |
| `--exploration_fraction` | Fraction of training steps dedicated to exploration. |
| `--tau` | Discount factor used for soft target updates. |
| `--nonoise` | Run with a noiseless simulator (ignore backend noise). |

---

## References
[1] Skolik, Andrea, Sofiene Jerbi, and Vedran Dunjko. "Quantum agents in the gym: a variational quantum algorithm for deep q-learning." Quantum 6 (2022): 720.

[2] Towers, Mark, et al. "Gymnasium: A standard interface for reinforcement learning environments." arXiv preprint arXiv:2407.17032 (2024).

[3] Kruse, Georg, et al. "Benchmarking quantum reinforcement learning." arXiv preprint arXiv:2502.04909 (2025).


