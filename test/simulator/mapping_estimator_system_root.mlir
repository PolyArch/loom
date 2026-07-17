// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --system-matrix-case dual-spatial-shared-memory --output %t.dir/system.mlir
// RUN: env LOOM_CC=%loom-cc LOOM_CXX=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt %python %S/../app/ir_runner.py --stage dfg --case byte_swap --build-root %t.dir
// RUN: loom-dfg-sim %t.dir/byte_swap/main_func.dfg.mlir --invocations 8 --arg 0=0 --arg 0=1 --arg 0=2 --arg 0=3 --arg 0=4 --arg 0=5 --arg 0=6 --arg 0=7 --memref 1=0,-1,305419896,287454020,-16777216,255,-1412567295,16909060 --memref 2=0,0,0,0,0,0,0,0 --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --workload byte_swap --output %t.dir/byte.dfg.json
// RUN: FileCheck %s --check-prefix=DFG < %t.dir/byte.dfg.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc1 --workload byte_swap --output %t.dir/byte.system.mapping.csv --artifact %t.dir/byte.system.mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/byte.system.mapping.json
// RUN: loom-mapping-estimate --mapping-artifact %t.dir/byte.system.mapping.json --hardware-mlir %t.dir/system.mlir --output %t.dir/byte.estimate.json
// RUN: FileCheck %s --check-prefix=ESTIMATE < %t.dir/byte.estimate.json
// RUN: sed 's/"selected_acc_core": "acc1"/"selected_acc_core": "acc0"/' %t.dir/byte.system.mapping.json > %t.dir/byte.bad-core.mapping.json
// RUN: not loom-mapping-estimate --mapping-artifact %t.dir/byte.bad-core.mapping.json --hardware-mlir %t.dir/system.mlir --output %t.dir/byte.bad-core.estimate.json 2>&1 | FileCheck %s --check-prefix=BAD-CORE

// MAPPING-DAG: "kind": "pnr_mapping"
// MAPPING-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// MAPPING-DAG: "hardware_root_kind": "fabric.system"
// MAPPING-DAG: "selected_acc_core": "acc1"
// MAPPING-DAG: "status": "pass"
// MAPPING-DAG: "unplaced_records": 0
// MAPPING-DAG: "unrouted_edges": 0
// MAPPING-DAG: "operation": "llvm.intr.bswap",
// MAPPING-DAG: "software": "llvm.intr.bswap#0"
// MAPPING-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.bswap#0.operand0",
// MAPPING-DAG: "status": "routed",

// ESTIMATE-DAG: "kind": "mapping_estimate_report"
// ESTIMATE-DAG: "workload": "byte_swap"
// ESTIMATE-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// ESTIMATE-DAG: "status": "pass"
// ESTIMATE-DAG: "total_cost_score": {{[1-9][0-9]*}}

// DFG-DAG: "kind": "dfg_sim_report"
// DFG-DAG: "workload": "byte_swap"
// DFG-DAG: "graph": "g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0"
// DFG-DAG: "status": "pass"
// DFG-DAG: "arg2"
// DFG-DAG: "i32:2018915346"

// BAD-CORE: mapping hardware system_dual_spatial_shared_memory_soc::acc1 does not match selected system core system_dual_spatial_shared_memory_soc::acc0
