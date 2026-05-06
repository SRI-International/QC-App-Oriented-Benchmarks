@echo off
setlocal

REM Run package_test_1 for all 16 benchmarks.
REM Uses package_test_2 for hamlib (includes observable test).
REM Comment out any lines below to skip specific benchmarks.

set "RUNTEST_1=%~dp0package_test_1.bat"
set "RUNTEST_2=%~dp0package_test_2.bat"

echo ============================================================
echo Running all benchmarks
echo ============================================================

REM Group 1 - Simple deterministic
cmd /c "%RUNTEST_1%" bernstein_vazirani bv_benchmark %*
cmd /c "%RUNTEST_1%" deutsch_jozsa dj_benchmark %*
cmd /c "%RUNTEST_1%" grovers grovers_benchmark %*
cmd /c "%RUNTEST_1%" hidden_shift hs_benchmark %*

REM Group 2 - Transform / estimation
cmd /c "%RUNTEST_1%" amplitude_estimation ae_benchmark %*
cmd /c "%RUNTEST_1%" phase_estimation pe_benchmark %*
cmd /c "%RUNTEST_1%" quantum_fourier_transform qft_benchmark %*

REM Group 3 - Complex (qiskit subdir)
cmd /c "%RUNTEST_1%" hydrogen_lattice hydrogen_lattice_benchmark %*
cmd /c "%RUNTEST_1%" maxcut maxcut_benchmark %*
rem cmd /c "%RUNTEST_1%" image_recognition image_recognition_benchmark %*
cmd /c "%RUNTEST_1%" hhl hhl_benchmark %*
cmd /c "%RUNTEST_1%" shors shors_benchmark %*
cmd /c "%RUNTEST_1%" vqe vqe_benchmark %*

REM Group 4 - Simulation
cmd /c "%RUNTEST_1%" hamiltonian_simulation hamiltonian_simulation_benchmark %*
cmd /c "%RUNTEST_1%" monte_carlo mc_benchmark %*

REM Hamlib uses package_test_2 (includes observable test)
cmd /c "%RUNTEST_2%" %*

echo ============================================================
echo All benchmarks complete.
echo ============================================================

endlocal
