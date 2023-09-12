import hydrogen_lattice_benchmark
import numpy as np
from scipy.optimize import minimize


def test_hydrogen_lattice_hartree_fock():
    for num_qubits in [2, 4, 6, 8, 10]:
        for radius in [0.75, 1.0, 1.25]:
            # Arguments specific to execution of single instance of the Hydrogen Lattice objective function
            hl_app_args = dict(
                num_qubits=num_qubits,
                radius=radius,
                parameter_mode=1,  # all thetas are equivalent with parameter_mode=1
                thetas_array=[0.0],  # all thetas are zero
                backend_id="statevector_simulator",
            )
            energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(
                **hl_app_args
            )
            assert np.isclose(energy, key_metrics["hf_energy"])


def test_h2_converges_to_exact():
    for radius in [0.75, 1.0, 1.25]:
        # Initial guess for the thetas_array
        initial_guess = [0.0]

        # Arguments for run_objective_function
        hl_app_args = {
            "num_qubits": 2,
            "radius": radius,
            "parameter_mode": 2,
            "backend_id": "statevector_simulator",
        }

        # Variable to hold key_metrics
        key_metrics_output = {}

        # Objective function to minimize
        def objective_func(x):
            energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(
                thetas_array=x, **hl_app_args
            )
            key_metrics_output.update(key_metrics)
            return energy

        # Run the optimization
        result = minimize(objective_func, initial_guess, method="L-BFGS-B")

        # Extract optimized thetas_array and minimum energy
        optimized_thetas_array = result.x
        minimum_energy = result.fun

        # FCI = DOCI for minimal basis H2
        assert np.isclose(minimum_energy, key_metrics_output["doci_energy"])
        assert np.isclose(minimum_energy, key_metrics_output["fci_energy"])
