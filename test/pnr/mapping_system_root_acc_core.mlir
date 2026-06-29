// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --system-matrix-case dual-spatial-shared-memory --output %t.dir/system.mlir
// RUN: env BUILD_DIR=%t.dir/byte_swap LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/byte_swap/dfg_check.sh
// RUN: bash %S/run_mapping_summary.sh --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc1 --workload byte_swap --output %t.dir/byte.acc1.mapping.csv --artifact %t.dir/byte.acc1.mapping.json
// RUN: FileCheck %s --check-prefix=ACC1-CSV < %t.dir/byte.acc1.mapping.csv
// RUN: FileCheck %s --check-prefix=ACC1-JSON < %t.dir/byte.acc1.mapping.json
// RUN: bash %S/run_mapping_summary.sh --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc0 --workload byte_swap --output %t.dir/byte.acc0.mapping.csv --artifact %t.dir/byte.acc0.mapping.json
// RUN: FileCheck %s --check-prefix=ACC0-CSV < %t.dir/byte.acc0.mapping.csv
// RUN: FileCheck %s --check-prefix=ACC0-JSON < %t.dir/byte.acc0.mapping.json
// RUN: %python %S/mapping_summary.py --dfg-mlir %s --graph unsupported_loop --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc1 --workload unsupported_loop --output %t.dir/unsupported.mapping.csv --artifact %t.dir/unsupported.mapping.json
// RUN: FileCheck %s --check-prefix=UNSUPPORTED-CSV < %t.dir/unsupported.mapping.csv
// RUN: FileCheck %s --check-prefix=UNSUPPORTED-JSON < %t.dir/unsupported.mapping.json
// RUN: not bash %S/run_mapping_summary.sh --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core missing_acc --workload byte_swap --output %t.dir/missing.csv --artifact %t.dir/missing.json 2>&1 | FileCheck %s --check-prefix=MISSING

// ACC1-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ACC1-CSV-NEXT: byte_swap,system_dual_spatial_shared_memory_soc::acc1,byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__system_dual_spatial_shared_memory_soc%3A%3Aacc1,4,4,0,0,pass,mapped software graph to fabric resources

// ACC1-JSON-DAG: "kind": "pnr_mapping"
// ACC1-JSON-DAG: "workload": "byte_swap"
// ACC1-JSON-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// ACC1-JSON-DAG: "hardware_root_kind": "fabric.system"
// ACC1-JSON-DAG: "hardware_system": "system_dual_spatial_shared_memory_soc"
// ACC1-JSON-DAG: "selected_acc_core": "acc1"
// ACC1-JSON-DAG: "spatialcore_template": "shared_vector_alu_adg"
// ACC1-JSON-DAG: "mapping_id": "byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__system_dual_spatial_shared_memory_soc%3A%3Aacc1"
// ACC1-JSON-DAG: "status": "pass"
// ACC1-JSON-DAG: "placed_records": 4
// ACC1-JSON-DAG: "routed_edges": 4
// ACC1-JSON-DAG: "source_endpoint": "shared_vector_alu_adg::mem.load#0.result0"
// ACC1-JSON-DAG: "sink_endpoint": "shared_vector_alu_adg::fabric.switch#0.operand0"
// ACC1-JSON-DAG: "segment_kind": "resource_edge"
// ACC1-JSON-DAG: "segment_kind": "module_path"
// ACC1-JSON-NOT: ".out"
// ACC1-JSON-NOT: ".in"

// ACC0-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// ACC0-CSV-NEXT: byte_swap,system_dual_spatial_shared_memory_soc::acc0,byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__system_dual_spatial_shared_memory_soc%3A%3Aacc0,4,2,2,0,fail,unrouted software edges lack Fabric ADG connectivity

// ACC0-JSON-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc0"
// ACC0-JSON-DAG: "hardware_system": "system_dual_spatial_shared_memory_soc"
// ACC0-JSON-DAG: "selected_acc_core": "acc0"
// ACC0-JSON-DAG: "spatialcore_template": "shared_reduction_adg"
// ACC0-JSON-DAG: "mapping_id": "byte_swap__g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0__system_dual_spatial_shared_memory_soc%3A%3Aacc0"
// ACC0-JSON-DAG: "status": "fail"

// UNSUPPORTED-CSV: workload,hardware,mapping_id,placed_records,routed_edges,unrouted_edges,unplaced_records,status,diagnostic
// UNSUPPORTED-CSV-NEXT: unsupported_loop,system_dual_spatial_shared_memory_soc::acc1,unsupported_loop__unsupported_loop__system_dual_spatial_shared_memory_soc%3A%3Aacc1,,,,,unsupported,unsupported PnR graph operation: llvm.intr.ctpop
// UNSUPPORTED-JSON-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// UNSUPPORTED-JSON-DAG: "hardware_root_kind": "fabric.system"
// UNSUPPORTED-JSON-DAG: "hardware_system": "system_dual_spatial_shared_memory_soc"
// UNSUPPORTED-JSON-DAG: "selected_acc_core": "acc1"
// UNSUPPORTED-JSON-DAG: "spatialcore_template": "shared_vector_alu_adg"
// UNSUPPORTED-JSON-DAG: "mapping_id": "unsupported_loop__unsupported_loop__system_dual_spatial_shared_memory_soc%3A%3Aacc1"
// UNSUPPORTED-JSON-DAG: "status": "unsupported"

// MISSING: system hardware system_dual_spatial_shared_memory_soc does not contain acc_core missing_acc

module {
  dataflow.graph.func private @unsupported_loop(%ctrl: none,
                                                %lhs: i32,
                                                %rhs: i32)
      -> (none, i32) {
    %pop = llvm.intr.ctpop(%lhs) : (i32) -> i32
    dataflow.graph.return %ctrl, %pop : none, i32
  }
}
