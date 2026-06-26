// RUN: rm -rf %t.dir
// RUN: env BUILD_DIR=%t.dir/pack_bits LOOM_CC=%loom-cc LOOM_RAISE=%loom-raise LOOM_LOWER=%loom-lower LOOM_RAISE_OPT=%loom-raise-opt bash %S/../app/pack_bits/dfg_check.sh
// RUN: mkdir -p %t.dir/reports
// RUN: env LOOM_DFG_SIM=loom-dfg-sim bash %S/run_app_reduction_dfg_sim.sh pack_bits %t.dir/pack_bits/main_func.dfg.mlir %t.dir/reports/pack_bits.report.json %t.dir/summary.csv
// RUN: FileCheck %s --check-prefix=PACK-BITS < %t.dir/reports/pack_bits.report.json

// PACK-BITS-DAG: "kind": "dfg_sim_report"
// PACK-BITS-DAG: "workload": "pack_bits"
// PACK-BITS-DAG: "graph": "g_t_pack_bits_kernel_red_0_0"
// PACK-BITS-DAG: "status": "pass"
// PACK-BITS-DAG: "event_count": 298
// PACK-BITS-DAG: "dynamic_work_items": 32
// PACK-BITS-DAG: "dataflow.load": 32
// PACK-BITS-DAG: "dataflow.store": 1
// PACK-BITS-DAG: "llvm.intr.umin": 1
// PACK-BITS-DAG: "final_memory_state": {
// PACK-BITS-DAG: "arg11": [
// PACK-BITS-DAG: "i32:-749385939"
// PACK-BITS-DAG: "final_outputs": [
// PACK-BITS-DAG: "none"
// PACK-BITS-DAG: "i64:32"
