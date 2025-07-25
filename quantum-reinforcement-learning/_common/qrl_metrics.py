'''
Quantum Fourier Transform Benchmark Program - Metrics for QRL
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''

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
        self.steps = 0                    # Total steps taken
        self.explore_steps = 0
        self.exploit_steps = 0
    
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
        Print all tracked QRL metrics and computed averages.
        """
        print("QRL Metrics:")
        print(f"  Circuit Evaluations: {self.circuit_evaluations}")
        print(f"  Gradient Evaluations: {self.gradient_evaluations}")
        print(f"  Environment Evaluations: {self.env_evals}")
        print(f"  Number of Successes: {self.num_success}")
        print(f"  Number of Episodes: {self.num_episodes}")
        print(f"  Total Steps: {self.steps}")
        print(f"  Explore Steps: {self.explore_steps}")
        print(f"  Exploit Steps: {self.exploit_steps}")
        # Print computed averages, handling division by zero
        avg_return_run = self.calculate_average_return_per_run() if self.steps > 0 else 0.0
        avg_return_episode = self.calculate_average_return_per_episode() if self.num_episodes > 0 else 0.0
        print(f"  Average Return per Run: {avg_return_run}")
        print(f"  Average Return per Episode: {avg_return_episode}")