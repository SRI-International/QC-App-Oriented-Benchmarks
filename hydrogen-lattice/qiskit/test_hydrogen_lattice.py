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
            energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(**hl_app_args)
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
            energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(thetas_array=x, **hl_app_args)
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


def test_all_ones_h4_h6_r1p0():
    reference_energy = [0.2889279419643874, 1.0303098036634282]  # for r=1.0 h4 and h6, computed with external code
    for idx, num_qubits in enumerate([4, 6]):
        hl_app_args = {
            "num_qubits": num_qubits,
            "radius": 1.0,
            "parameter_mode": 2,
            "thetas_array": [1.0],
            "backend_id": "statevector_simulator",
        }
        energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(**hl_app_args)

        assert np.isclose(energy, reference_energy[idx])


def test_h4_preoptimized():
    hl_app_args = {
        "num_qubits": 4,
        "radius": 1.0,
        "parameter_mode": 2,
        "thetas_array": [0.06130415054606972, 0.03782678831535015, 0.15185290712860083, 0.03900390773961034],
        "backend_id": "statevector_simulator",
    }
    energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(**hl_app_args)

    assert np.isclose(energy, key_metrics["doci_energy"], atol=1e-5)
    assert np.isclose(energy, -2.133963877142266)


def test_h8_preoptimized():
    hl_app_args = {
        "num_qubits": 8,
        "radius": 1.0,
        "parameter_mode": 2,
        "thetas_array": [
            0.031881089808034724,
            0.015313387273404292,
            0.011207391278269674,
            0.015081079851112724,
            0.034590389008954406,
            0.023061020973892822,
            0.019079399843714314,
            0.00891871780735226,
            0.05128137578870619,
            0.039075999703602884,
            0.0180555186837568,
            0.009897282263869366,
            0.1312574256307422,
            0.043158222503003975,
            0.023816061083747845,
            0.01760223626449797,
        ],
        "backend_id": "statevector_simulator",
    }
    energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(**hl_app_args)

    assert np.isclose(energy, key_metrics["doci_energy"], atol=1e-5)
    assert np.isclose(energy, -4.2090835937969215)


def test_h6_preoptimized():
    hl_app_args = {
        "num_qubits": 6,
        "radius": 1.0,
        "parameter_mode": 2,
        "thetas_array": [
            0.04149555611957753,
            0.018494869732410688,
            0.021886977480671397,
            0.0533855896292699,
            0.040449656324743476,
            0.013187049613030346,
            0.14056216305071437,
            0.04126203433294259,
            0.023822028819589253,
        ],
        "backend_id": "statevector_simulator",
    }
    energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(**hl_app_args)

    assert np.isclose(energy, key_metrics["doci_energy"], atol=1e-5)
    assert np.isclose(energy, -3.17083762)


def test_h10_preoptimized():
    hl_app_args = {
        "num_qubits": 10,
        "radius": 1.0,
        "parameter_mode": 2,
        "thetas_array": [
            0.0260949460242973,
            0.01291594480660682,
            0.010376626572819824,
            0.007722266613889202,
            0.011447734408605054,
            0.025699053226966855,
            0.01902316437411572,
            0.010944697406789172,
            0.013466689550371053,
            0.006647678079634941,
            0.03326879398580485,
            0.022639871347627512,
            0.02110942996583013,
            0.00904952602114243,
            0.00727764656053018,
            0.049980276038360784,
            0.03867660198798731,
            0.01884531294758396,
            0.013475156471301155,
            0.007938315577706722,
            0.12250812566405145,
            0.04399160690835546,
            0.025442725392618753,
            0.016452600953270852,
            0.014217086338594176,
        ],
        "backend_id": "statevector_simulator",
    }
    energy, key_metrics = hydrogen_lattice_benchmark.run_objective_function(**hl_app_args)

    assert np.isclose(energy, key_metrics["doci_energy"], atol=1e-5)
    assert np.isclose(energy, -5.24830435)


def test_h6_full_opt():
    def L_BFGS_B(objective_function, initial_parameters, callback):
        ret = minimize(objective_function, x0=initial_parameters, method="l-bfgs-b", tol=1e-8)

        return ret

    # Arguments specific to Hydrogen Lattice benchmark method (2)
    hl_app_args = dict(
        min_qubits=6,
        max_qubits=6,  # do h6
        method=2,
        backend_id="statevector_simulator",
        radius=1.25,  # select single problem radius, None = use max_circuits problems
        parameter_mode=2,  # 1 - use single theta parameter, 2 - map multiple thetas to pairs
        max_circuits=1,  # just run once
        minimizer_function=L_BFGS_B,
        # disable display options for line plots
        line_y_metrics=None,
        line_x_metrics=None,
        # disable display options for bar plots
        bar_y_metrics=None,
        bar_x_metrics=None,
        # disable display options for area plots
        score_metric=None,
        x_metric=None,
    )

    # Run the benchmark in method 2
    hydrogen_lattice_benchmark.run(**hl_app_args)

    # Get final results from last run
    energy, key_metrics = hydrogen_lattice_benchmark.get_final_results()

    assert np.isclose(energy, key_metrics["doci_energy"], atol=1e-4)
    assert np.isclose(energy, -3.01129635)
