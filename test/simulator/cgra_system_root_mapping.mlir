// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --system-matrix-case dual-spatial-shared-memory --output %t.dir/system.mlir
// RUN: env LOOM_CC=%loom-cc LOOM_CXX=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt %python %S/../app/ir_runner.py --stage dfg --case byte_swap --build-root %t.dir
// RUN: loom-dfg-sim %t.dir/byte_swap/main_func.dfg.mlir --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 0=none --arg 0=none --memref 1=0,-1,305419896,287454020,-16777216,255,-1412567295,16909060 --memref 2=0,0,0,0,0,0,0,0 --arg 3=0 --arg 3=1 --arg 3=2 --arg 3=3 --arg 3=4 --arg 3=5 --arg 3=6 --arg 3=7 --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --workload byte_swap --output %t.dir/byte.dfg.json
// RUN: FileCheck %s --check-prefix=DFG < %t.dir/byte.dfg.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc1 --workload byte_swap --output %t.dir/byte.system.mapping.csv --artifact %t.dir/byte.system.mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/byte.system.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/byte.dfg.json --mapping-artifact %t.dir/byte.system.mapping.json --hardware-mlir %t.dir/system.mlir --output %t.dir/byte.cgra.json
// RUN: FileCheck %s --check-prefix=CGRA < %t.dir/byte.cgra.json
// RUN: loom-sim-cycle-summary --dfg-report %t.dir/byte.dfg.json --cgra-report %t.dir/byte.cgra.json --output %t.dir/byte.summary.csv
// RUN: FileCheck %s --check-prefix=SUMMARY < %t.dir/byte.summary.csv
// RUN: sed 's/"selected_acc_core": "acc1"/"selected_acc_core": "acc0"/' %t.dir/byte.system.mapping.json > %t.dir/byte.bad-core.mapping.json
// RUN: not loom-cgra-sim --dfg-report %t.dir/byte.dfg.json --mapping-artifact %t.dir/byte.bad-core.mapping.json --hardware-mlir %t.dir/system.mlir --output %t.dir/byte.bad-core.cgra.json 2>&1 | FileCheck %s --check-prefix=BAD-CORE

// MAPPING-DAG: "kind": "pnr_mapping"
// MAPPING-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// MAPPING-DAG: "status": "pass"

// CGRA-DAG: "kind": "cgra_sim_report"
// CGRA-DAG: "workload": "byte_swap"
// CGRA-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// CGRA-DAG: "status": "pass"
// CGRA-DAG: "functional_state_source": "carried_from_dfg_sim_report"
// CGRA-DAG: "arg2"
// CGRA-DAG: "i32:2018915346"

// DFG-DAG: "kind": "dfg_sim_report"
// DFG-DAG: "workload": "byte_swap"
// DFG-DAG: "graph": "g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0"
// DFG-DAG: "status": "pass"
// DFG-DAG: "arg2"
// DFG-DAG: "i32:2018915346"

// SUMMARY: kernel,dfg_sim_cycles,cgra_sim_cycles,status,diagnostic
// SUMMARY-NEXT: byte_swap,{{[1-9][0-9]*}},{{[1-9][0-9]*}},pass,

// BAD-CORE: mapping hardware system_dual_spatial_shared_memory_soc::acc1 does not match selected system core system_dual_spatial_shared_memory_soc::acc0
