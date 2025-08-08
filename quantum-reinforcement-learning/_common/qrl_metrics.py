'''
Quantum Fourier Transform Benchmark Program - Metrics for QRL
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''
import matplotlib.pyplot as plt

class qrl_metrics:
    """
    Class to track and compute metrics for Quantum Reinforcement Learning (QRL) benchmarks.
    """

    def __init__(self):
        self.circuit_evaluations = 0      # Number of quantum circuit evaluations performed
        self.gradient_evaluations = 0     # Number of gradient evaluations performed
        self.env_evals = 0                # Number of environment evaluations (sample, step, reset)
        self.num_success = 0              # Number of successful episodes or tasks completed
        self.num_episodes = 0             # Total number of episodes run
        self.steps = 0                    # Total steps taken across all episodes
        self.explore_steps = 0            # Number of steps taken during exploration phase
        self.exploit_steps = 0            # Number of steps taken during exploitation phase
        self.total_time = 0               # Total wall-clock time for the benchmark (seconds)
        self.quantum_time = 0             # Time spent on quantum circuit execution (seconds)
        self.step_time = 0                # Time spent on a step (seconds)
        self.environment_time = 0         # Time spent interacting with the environment (seconds)
        self.gradient_time = 0            # Time spent computing gradients (seconds)
        self.loss_history = []            # History of loss
        self.step_history = []
    
    def calculate_average_return_per_run(self):
        """
        Calculate the average return per run (successes per step).
        Returns:
            float: The average return per run.
        """
        self.average_return = self.num_success / self.steps  
        return self.average_return

    def calculate_average_return_per_episode(self):
        """
        Calculate the average return per episode (successes per episode).
        Returns:
            float: The average return per episode.
        """
        return self.num_success / self.num_episodes 
    
    def print_metrics(self):
        """
        Print all tracked QRL metrics, computed averages, and timing information.
        """
        print("============ QRL Metrics:")
        print(f"  Circuit Evaluations: {self.circuit_evaluations}")
        print(f"  Gradient Evaluations: {self.gradient_evaluations}")
        print(f"  Environment Evaluations: {self.env_evals}")
        print(f"  Number of Successes: {self.num_success}")
        print(f"  Number of Episodes: {self.num_episodes}")
        print(f"  Total Steps: {self.steps}")
        print(f"  Explore Steps: {self.explore_steps}")
        print(f"  Exploit Steps: {self.exploit_steps}")
        # Print computed averages, handling division by zero
        avg_return_run = self.calculate_average_return_per_run() 
        avg_return_episode = self.calculate_average_return_per_episode() if self.num_episodes > 0 else 0.0
        print(f"  Average Return per Run: {avg_return_run}")
        print(f"  Average Return per Episode: {avg_return_episode}")
        # Print timing information
        print(f"  Total Time: {self.total_time:.4f} seconds")
        print(f"  Quantum Time: {self.quantum_time:.4f} seconds")
        print(f"  Step Time: {self.step_time:.4f} seconds")
        print(f"  Environment Time: {self.environment_time:.4f} seconds")
        print(f"  Gradient Time: {self.gradient_time:.4f} seconds")
        avg_step_time = self.step_time / self.steps
        print(f"  Average Step Time: {avg_step_time:.4f} seconds", flush = True)

    def update_history(self):
        self.step_history.append([self.circuit_evaluations, self.gradient_evaluations, self.num_episodes, self.num_success, self.env_evals, self.explore_steps, self.exploit_steps])

    def plot_metrics(self):
        circuit_evals = [row[0] for row in self.step_history]
        gradient_evals = [row[1] for row in self.step_history]
        num_episodes = [row[2] for row in self.step_history]
        num_success = [row[3] for row in self.step_history]
        env_evals = [row[4] for row in self.step_history]
        explores = [row[5] for row in self.step_history]
        exploits = [row[6] for row in self.step_history]

        # Create x-axis as step index
        x = list(range(1, len(self.step_history) + 1))

        # Plot each metric in its own subplot with shared x-axis
        fig, axs = plt.subplots(7, 1, figsize=(8, 12), sharex=True)

        axs[0].plot(x, circuit_evals, marker='o')
        axs[0].set_ylabel('Circuit Evals')
        axs[0].set_ylim(bottom=0)

        axs[1].plot(x, gradient_evals, marker='o')
        axs[1].set_ylabel('Gradient Evals')

        axs[2].plot(x, num_episodes, marker='o')
        axs[2].set_ylabel('Episodes')

        axs[3].plot(x, num_success, marker='o')
        axs[3].set_ylabel('Successes')
        axs[3].set_ylim(bottom=0)

        axs[4].plot(x, env_evals, marker='o')
        axs[4].set_ylabel('Env Evals')

        axs[5].plot(x, explores, marker='o')
        axs[5].set_ylabel('Explore steps')
        axs[5].set_ylim(bottom=0)

        axs[6].plot(x, exploits, marker='o')
        axs[6].set_ylabel('Exploit steps')
        axs[6].set_xlabel('Step Index')

        fig.suptitle('Metrics Over Steps')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig("stephistory.png")
