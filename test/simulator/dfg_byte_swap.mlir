// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: env LOOM_CC=%loom-cc LOOM_CXX=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt %python %S/../app/ir_runner.py --stage dfg --case byte_swap --build-root %t.dir
// RUN: loom-dfg-sim %t.dir/byte_swap/main_func.dfg.mlir --invocations 8 --arg 0=0 --arg 0=1 --arg 0=2 --arg 0=3 --arg 0=4 --arg 0=5 --arg 0=6 --arg 0=7 --memref 1=0,-1,305419896,287454020,-16777216,255,-1412567295,16909060 --memref 2=0,0,0,0,0,0,0,0 --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --workload byte_swap --output %t.dir/byte.dfg.json
// RUN: FileCheck %s --check-prefix=DFG < %t.dir/byte.dfg.json

// DFG-DAG: "kind": "dfg_sim_report"
// DFG-DAG: "workload": "byte_swap"
// DFG-DAG: "graph": "g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0"
// DFG-DAG: "status": "pass"
// DFG-DAG: "arg2"
// DFG-DAG: "i32:2018915346"
