'''
Quantum Fourier Transform Benchmark Program - Metrics for QRL
(C) Quantum Economic Development Consortium (QED-C) 2025.
'''
import matplotlib.pyplot as plt
import numpy as np

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
        """
        step_history: list of lists like
        [circuit_evals, gradient_evals, num_episodes, num_success, env_evals]
        appended once per step.
        """
        A = np.asarray(self.step_history, dtype=float)
        if A.ndim != 2 or A.shape[1] < 1:
            raise ValueError("step_history must be a list of lists with at least 1 column.")

        x = np.arange(len(A))  # steps: 0..n-1

        # Extract columns
        circuit_cum = A[:, 0]     # cumulative circuit evaluations
        grad_cum    = A[:, 1]     # cumulative gradient evaluations
        env_evals   = A[:, 4]     # environment evaluations
        explore_raw = A[:, 5]     # explore steps
        exploit_raw = A[:, 6]     # exploit steps

        # Helper: convert cumulative -> per-step
        def per_step(series):
            return np.diff(np.concatenate(([0.0], series)))

        # Always treat circuit evals as cumulative
        circuit_per_step = per_step(circuit_cum)

        # Detect if explore/exploit are cumulative (non-decreasing)
        explore_per_step = per_step(explore_raw) if np.all(np.diff(explore_raw) >= 0) else explore_raw
        exploit_per_step = per_step(exploit_raw) if np.all(np.diff(exploit_raw) >= 0) else exploit_raw

        # Prepare plots
        plots = [
            ("Circuit evals (per step)", circuit_per_step, "Circuit evals / step"),
            ("Explore steps (per step)", explore_per_step, "Explore / step"),
            ("Exploit steps (per step)", exploit_per_step, "Exploit / step"),
            #("Gradient evals (cumulative)", grad_cum, "Grad evals (cum)"),
            ("Env evals", env_evals, "Env evals"),
        ]

        fig, axes = plt.subplots(len(plots), 1, figsize=(10, 2*len(plots)), sharex=True)

        for ax, (title, y, ylabel) in zip(axes, plots):
            ax.plot(x, y, linewidth=0.5, marker='x')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, linestyle='--', alpha=0.5)
            if ylabel == "Circuit evals / step":
                ax.set_ylim(0, 2) 

        axes[-1].set_xlabel("Step")
        fig.tight_layout()
        plt.savefig("step_history.svg", dpi=600)
        plt.show()
    
    def print_final_metrics(self):
        """
        Pretty-print QRL metrics in a neat table format, grouped by category.
        - Totals
        - Averages
        - Timing
        """
        avg_return_run = self.calculate_average_return_per_run()
        avg_return_episode = self.calculate_average_return_per_episode()
        avg_step_time = (self.step_time / self.steps) if self.steps > 0 else 0.0

        # Grouped metrics
        totals = [
            ("Circuit Evaluations", self.circuit_evaluations),
            ("Gradient Evaluations", self.gradient_evaluations),
            ("Environment Evaluations", self.env_evals),
            ("Number of Successes", self.num_success),
            ("Number of Episodes", self.num_episodes),
            ("Total Steps", self.steps),
            ("Explore Steps", self.explore_steps),
            ("Exploit Steps", self.exploit_steps),
        ]

        averages = [
            ("Average Return per Run", f"{avg_return_run:.4f}"),
            ("Average Return per Episode", f"{avg_return_episode:.4f}"),
            ("Average Step Time (s)", f"{avg_step_time:.4f}"),
        ]

        timings = [
            ("Total Time (s)", f"{self.total_time:.4f}"),
            ("Quantum Time (s)", f"{self.quantum_time:.4f}"),
            ("Environment Time (s)", f"{self.environment_time:.4f}"),
            ("Gradient Time (s)", f"{self.gradient_time:.4f}"),
        ]

        all_metrics = [("TOTALS", totals), ("AVERAGES", averages), ("TIMINGS", timings)]

        # Determine column widths
        max_key_len = max(len(k) for _, group in all_metrics for k, _ in group)
        max_val_len = max(len(str(v)) for _, group in all_metrics for _, v in group)
        line = "═" * (max_key_len + max_val_len + 7)

        # Print table
        print("╔" + line + "╗")
        print("║{:^{width}}║".format(" QRL Benchmark Metrics ", width=max_key_len + max_val_len + 7))
        print("╠" + line + "╣")

        for section, group in all_metrics:
            # Section header
            print("║ {:^{width}} ║".format(section, width=max_key_len + max_val_len + 5))
            print("╟" + line + "╢")
            # Section rows
            for key, val in group:
                print(f"║ {key:<{max_key_len}} │ {str(val):>{max_val_len}} ║")
            print("╠" + line + "╣")

        print("╚" + line + "╝")
