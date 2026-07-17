// RUN: loom-dfg-sim %s --graph sum_ptr_load --arg 0=none --arg 0=none --arg 0=none --arg 1=0 --arg 2=3 --arg 3=1 --memref 4=2.000000e+00,4.000000e+00,8.000000e+00 --arg 5=0.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "sum_ptr_load"
// CHECK-DAG: "graph": "sum_ptr_load"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// CHECK-DAG: "dataflow.load": 3
// CHECK-DAG: "dataflow.carry": 5
// CHECK-DAG: "dataflow.gate": 4
// CHECK-DAG: "dataflow.demux": 4
// CHECK-DAG: "f32:14"

module {
  llvm.func @external_decl(!llvm.ptr) -> i32

  func.func private @host_stub() {
    return
  }

  dataflow.graph.func private @sum_ptr_load(%ctrl: none, %lb: i64, %ub: i64,
                                            %step: i64, %ptr: !llvm.ptr,
                                            %init: f32) -> (none, f32) {
    %mem = builtin.unrealized_conversion_cast %ptr : !llvm.ptr to memref<?xf32>
    %iv, %phase = dataflow.stream %lb, %ub, %step
        {step_op = "+=", cont_cond = "<"} : i64
    %idx = arith.index_cast %iv : i64 to index
    %data, %done = dataflow.load %mem[%idx] %ctrl : memref<?xf32>
    %carry = dataflow.carry %phase, %init, %next : f32
    %body_phase, %body_carry = dataflow.gate %phase, %carry : f32
    %exit:2 = dataflow.demux %phase, %carry : (i1, f32) -> (f32, f32)
    %next = arith.addf %body_carry, %data : f32
    %done_sync = dataflow.sync %done : (none) -> none
    dataflow.graph.return %done_sync, %exit#0 : none, f32
  }
}
