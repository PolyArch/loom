// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: echo '{"schema_version":1,"kind":"dfg_sim_report","workload":"cfft_red3_fmul_pair","graph":"cfft_red3_fmul_pair","status":"pass","operation_semantics_source":"loom.sim.operation_semantics.v1","operation_cost_model_source":"loom.sim.operation_cost.v1","optimistic_cycles":7,"final_outputs":["none"],"final_memory_state":{"arg5":["f32:0.500000","f32:-1.000000","f32:1.500000","f32:-2.000000"]}}' > %t.dir/dfg.json
// RUN: loom-pnr-map --dfg-mlir %S/../pnr/mapping_mem_route.mlir --graph cfft_red3_fmul_pair --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --hardware shared_reduction_adg --workload cfft_red3_fmul_pair --output %t.dir/mapping.csv --artifact %t.dir/mapping.json
// RUN: FileCheck %s --check-prefix=MAPPING < %t.dir/mapping.json
// RUN: loom-cgra-sim --dfg-report %t.dir/dfg.json --mapping-artifact %t.dir/mapping.json --hardware-mlir %S/../pnr/shared_reduction_adg.mlir --output %t.dir/cgra.json
// RUN: FileCheck %s --check-prefix=CGRA < %t.dir/cgra.json

// MAPPING-DAG: "status": "pass"
// MAPPING-DAG: "operation": "llvm.store"
// MAPPING-DAG: "resource_kind": "fabric.mem.store"
// MAPPING-DAG: "edge_ref": "arith.mulf#1.result0->llvm.store#0.operand0"
// MAPPING-DAG: "sink_endpoint": "shared_reduction_adg::mem.store#1.operand1"

// CGRA-DAG: "kind": "cgra_sim_report"
// CGRA-DAG: "workload": "cfft_red3_fmul_pair"
// CGRA-DAG: "status": "pass"
// CGRA-DAG: "mapping_id": "cfft_red3_fmul_pair__cfft_red3_fmul_pair__shared_reduction_adg"
