#!/bin/bash

cp maxcut_benchmark.py mxb_cp.py
cp ../../_common/qiskit/execute.py ex_cp.py

cat ../_common/common.py ../../_common/metrics.py ex_cp.py mxb_cp.py > catted.py
sed -i -f script.sed catted.py
cat catted.py add_main.py > maxcut_runtime.py

rm mxb_cp.py ex_cp.py catted.py

