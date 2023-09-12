import hydrogen_lattice_benchmark
import numpy as np


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
            energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(**hl_app_args)
            assert np.isclose(energy, key_metrics["hf_energy"])
