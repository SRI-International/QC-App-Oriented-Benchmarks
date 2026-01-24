setlocal

echo Testing package implementation with Hamlib Benchmark
echo

cd ..

echo ... run in BM directory ... 
cd hamlib
call python hamlib_simulation_benchmark.py -nop -nod
cd ..

pause

echo ... run at top level ... 
call python hamlib/hamlib_simulation_benchmark.py -nop -nod

pause

echo ... run at top level as a module 
call python -m hamlib.hamlib_simulation_benchmark -nop -nod

pause

echo ... run at top level as a module (observable test)
call python -m hamlib.hamlib_simulation_benchmark -nop -nod -obs

endlocal
