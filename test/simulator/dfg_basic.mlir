// RUN: loom-dfg-sim %s --graph sum4 --arg 0=0 --arg 1=4 --arg 2=1 --arg 3=0.000000e+00 --arg 4=1.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "schema_version": "2.2"
// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "sum4"
// CHECK-DAG: "graph": "sum4"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// CHECK-DAG: "operation_semantics_source": "loom.sim.operation_semantics.v1"
// CHECK-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// CHECK-DAG: "final_outputs":
// CHECK-DAG: "none",
// CHECK-DAG: "f32:4"
// CHECK-NOT: cycles

module {
  dataflow.graph.func private @sum4(%ctrl: none, %lb: i64, %ub: i64,
    %step: i64, %init: f32, %increment: f32)
      -> (none, f32) {
    %iv, %phase = dataflow.stream %lb, %ub, %step
        step add while slt : i64
    %carry = dataflow.carry %phase, %init, %next : f32
    %body_phase, %body_carry = dataflow.gate %phase, %carry : f32
    %exit:2 = dataflow.demux %phase, %carry : (i1, f32) -> (f32, f32)
    %increment_raw = dataflow.invariant %phase, %increment : f32
    %increment_phase, %body_increment =
        dataflow.gate %phase, %increment_raw : f32
    %next = arith.addf %body_carry, %body_increment : f32
    %published:2 = dataflow.sync %ctrl, %exit#0
        : (none, f32) -> (none, f32)
    dataflow.graph.return %published#0, %published#1 : none, f32
  }
}
