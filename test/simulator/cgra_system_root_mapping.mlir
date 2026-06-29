// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --system-matrix-case dual-spatial-shared-memory --output %t.dir/system.mlir
// RUN: env BUILD_DIR=%t.dir/byte_swap LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/byte_swap/dfg_check.sh
// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"byte_swap","graph":"g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":11,"final_outputs":[],"final_memory_state":{}}' > %t.dir/byte.dfg.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc1 --workload byte_swap --output %t.dir/byte.system.mapping.csv --artifact %t.dir/byte.system.mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/byte.dfg.json --mapping-artifact %t.dir/byte.system.mapping.json --hardware-mlir %t.dir/system.mlir --output %t.dir/byte.cgra.json
// RUN: FileCheck %s --check-prefix=CGRA < %t.dir/byte.cgra.json
// RUN: sed 's/"selected_acc_core": "acc1"/"selected_acc_core": "acc0"/' %t.dir/byte.system.mapping.json > %t.dir/byte.bad-core.mapping.json
// RUN: not loom-cgra-sim --dfg-report %t.dir/byte.dfg.json --mapping-artifact %t.dir/byte.bad-core.mapping.json --hardware-mlir %t.dir/system.mlir --output %t.dir/byte.bad-core.cgra.json 2>&1 | FileCheck %s --check-prefix=BAD-CORE

// CGRA-DAG: "kind": "cgra_sim_report"
// CGRA-DAG: "workload": "byte_swap"
// CGRA-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// CGRA-DAG: "mapping_id": "byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__system_dual_spatial_shared_memory_soc%3A%3Aacc1"
// CGRA-DAG: "status": "pass"
// CGRA-DAG: "routed_edges": 4
// CGRA-DAG: "route_segments": 14

// BAD-CORE: mapping hardware system_dual_spatial_shared_memory_soc::acc1 does not match selected system core system_dual_spatial_shared_memory_soc::acc0
