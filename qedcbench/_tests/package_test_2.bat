@echo off
setlocal enabledelayedexpansion

REM Navigate to the benchmark root (parent of _tests)
pushd "%~dp0.."

REM Args:
REM - 0 args -> defaults (hamlib, -a qiskit)
REM - args starting with "-" -> hamlib + all args passed to Python
REM
REM Examples:
REM   package_test_2.bat
REM   package_test_2.bat -a cudaq
REM   package_test_2.bat -a qiskit --num_qubits 6

set folder=hamlib
set bmname=hamlib_simulation_benchmark

if "%~1"=="" (
    set extra_args=-a qiskit
) else (
    set extra_args=%*
)

echo ============================================================
echo Testing: %folder% / %bmname%
echo ============================================================
echo extra_args=-nop -nod %extra_args%

echo ... run in BM directory ...
pushd %folder%
call python %bmname%.py -nop -nod %extra_args%
popd

rem pause

echo ... run at top level ...
call python %folder%/%bmname%.py -nop -nod %extra_args%

rem pause

echo ... run at top level as a module
call python -m %folder%.%bmname% -nop -nod %extra_args%

rem pause

echo ... run at top level as a module (observable test)
call python -m %folder%.%bmname% -nop -nod %extra_args% -obs

popd
endlocal
