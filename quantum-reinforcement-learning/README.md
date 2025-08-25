# Quantum Reinforcement Learning (QRL) – Benchmark Program

---

**Note:** This repository is a work in progress (WIP), including this README, and will be updated over time.  

With the advent of quantum computers, **Quantum Machine Learning (QML)** has emerged as a key application area within the quantum computing ecosystem. Among its subfields, **Quantum Reinforcement Learning (QRL)** is rapidly gaining attention. QRL typically employs a hybrid quantum–classical loop, where a separate classical module—called the *environment*—interacts with the quantum agent.  

In this work, we provide a **Quantum Deep Q-Network (DQN)** implementation to solve the **FrozenLake** environment from the [Gymnasium](https://gymnasium.farama.org/) library.  

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
