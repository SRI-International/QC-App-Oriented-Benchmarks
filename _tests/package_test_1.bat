setlocal

echo "============================================================"
echo Testing package implementation (default to Hidden Shift Benchmark)
echo

cd ..

@echo off
if "%2"=="" (
    if not "%1"=="" (
        echo Error: Both arguments required or none
        exit /b 1
    )
    set folder=hidden_shift
    set bmname=hs_benchmark
) else (
    set folder=%1
    set bmname=%2
)

REM Now use %folder% and %bmname%
echo folder=%folder%
echo bmname=%bmname%


echo ... run in BM directory ... 
cd %folder%
call python %bmname%.py
cd ..

pause

echo ... run at top level ... 
call python %folder%/%bmname%.py

pause

echo ... run at top level as a module 
call python -m %folder%.%bmname%

endlocal
