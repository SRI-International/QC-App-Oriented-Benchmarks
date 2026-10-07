@echo off
setlocal enabledelayedexpansion

REM Navigate to the benchmark root (parent of _tests)
pushd "%~dp0.."

REM Args:
REM - 0 args -> defaults (hidden_shift / hs_benchmark), no extra args
REM - first arg starts with "-" -> defaults + all args passed to Python
REM - 2+ args (first not starting with "-") -> folder/bmname + rest to Python
REM
REM Examples:
REM   package_test_1.bat
REM   package_test_1.bat -a cudaq
REM   package_test_1.bat hydrogen_lattice hydrogen_lattice_benchmark
REM   package_test_1.bat hydrogen_lattice hydrogen_lattice_benchmark -a qiskit

set folder=hidden_shift
set bmname=hs_benchmark
set extra_args=

if "%~1"=="" goto :args_done

REM Check if first arg starts with "-"
set "first=%~1"
if not "!first:~0,1!"=="-" goto :parse_folder_bmname
set extra_args=%*
goto :args_done

:parse_folder_bmname
if "%~2"=="" goto :arg_error
set folder=%~1
set bmname=%~2
shift
shift
set extra_args=

:collect_args
if "%~1"=="" goto :args_done
set extra_args=!extra_args! %1
shift
goto :collect_args

:args_done

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

popd
endlocal
goto :eof

:arg_error
echo Error: Need 0 args, flags only, or 2+ args (folder bmname [python_args...])
popd
endlocal
exit /b 1
