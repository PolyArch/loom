// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --system-matrix-case dual-spatial-shared-memory --output %t.dir/system.mlir
// RUN: env LOOM_CC=%loom-cc LOOM_CXX=%loom-c++ LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt %python %S/../app/ir_runner.py --stage dfg --case byte_swap --build-root %t.dir
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc1 --workload byte_swap --output %t.dir/byte.acc1.mapping.csv --artifact %t.dir/byte.acc1.mapping.json
// RUN: FileCheck %s --check-prefix=ACC1 < %t.dir/byte.acc1.mapping.json
// RUN: loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc0 --workload byte_swap --output %t.dir/byte.acc0.mapping.csv --artifact %t.dir/byte.acc0.mapping.json
// RUN: FileCheck %s --check-prefix=ACC0 < %t.dir/byte.acc0.mapping.json
// RUN: not loom-pnr-map --dfg-mlir %s --graph unsupported_loop --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core acc1 --workload unsupported_loop --output %t.dir/unsupported.mapping.csv --artifact %t.dir/unsupported.mapping.json 2>&1 | FileCheck %s --check-prefix=UNSUPPORTED-DIAG
// RUN: not loom-pnr-map --dfg-mlir %t.dir/byte_swap/main_func.dfg.mlir --graph g_t__ZN12_GLOBAL__N_119byte_swap_candidateEPKjPjj_0_0 --hardware-mlir %t.dir/system.mlir --hardware system_dual_spatial_shared_memory_soc --hardware-root-kind system --acc-core missing_acc --workload byte_swap --output %t.dir/missing.csv --artifact %t.dir/missing.json 2>&1 | FileCheck %s --check-prefix=MISSING

// ACC1-DAG: "kind": "pnr_mapping"
// ACC1-DAG: "workload": "byte_swap"
// ACC1-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc1"
// ACC1-DAG: "hardware_root_kind": "fabric.system"
// ACC1-DAG: "hardware_system": "system_dual_spatial_shared_memory_soc"
// ACC1-DAG: "selected_acc_core": "acc1"
// ACC1-DAG: "spatialcore_template": "shared_vector_alu_adg"
// ACC1-DAG: "status": "pass"
// ACC1-DAG: "unplaced_records": 0
// ACC1-DAG: "unrouted_edges": 0
// ACC1-DAG: "operation": "llvm.intr.bswap",
// ACC1-DAG: "software": "llvm.intr.bswap#0"
// ACC1-DAG: "edge_ref": "dataflow.load#0.result0->llvm.intr.bswap#0.operand0",
// ACC1-DAG: "status": "routed",

// ACC0-DAG: "hardware": "system_dual_spatial_shared_memory_soc::acc0"
// ACC0-DAG: "hardware_system": "system_dual_spatial_shared_memory_soc"
// ACC0-DAG: "selected_acc_core": "acc0"
// ACC0-DAG: "spatialcore_template": "shared_reduction_adg"
// ACC0-DAG: "status": "fail"

// UNSUPPORTED-DIAG: finalized graph contains unregistered actor 'llvm.intr.ctpop'
// MISSING: system hardware system_dual_spatial_shared_memory_soc does not contain acc_core missing_acc

module {
  dataflow.graph private @unsupported_loop(%ctrl: none,
                                                %lhs: i32,
                                                %rhs: i32)
      -> (i32)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %pop = llvm.intr.ctpop(%lhs) : (i32) -> i32
    %published:2 = dataflow.sync %ctrl, %pop
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
}
