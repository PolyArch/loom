// RUN: loom-dfg-sim %s --graph sum4 --arg 0=none --arg 1=0 --arg 2=4 --arg 3=1 --arg 4=0.000000e+00 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "schema_version": "2.1"
// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "sum4"
// CHECK-DAG: "graph": "sum4"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "metric_definition": "weighted_operations_plus_library_work_diversity_and_address.v1"
// CHECK-DAG: "operation_semantics_source": "loom.sim.operation_semantics.v1"
// CHECK-DAG: "operation_cost_model_source": "loom.sim.operation_cost.v1"
// CHECK-DAG: "operation_cost_score": 37
// CHECK-DAG: "weighted_operation_score": 32
// CHECK-DAG: "operation_diversity_score": 5
// CHECK-DAG: "wavefront_steps": 12
// CHECK-DAG: "event_count": 27
// CHECK-DAG: "final_outputs":
// CHECK-DAG: "none",
// CHECK-DAG: "f32:4"
// CHECK-NOT: cycles

module {
  dataflow.graph.func private @sum4(%ctrl: none, %lb: i64, %ub: i64,
                                    %step: i64, %init: f32)
      -> (none, f32) {
    %one = dataflow.constant %ctrl {const_value = 1.000000e+00 : f32} : f32
    %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "+=", cont_cond = "<"} : i64
    %inc = dataflow.invariant %rwc, %one : f32
    %carry = dataflow.carry %rwc, %init, %next : f32
    %next = arith.addf %carry, %inc : f32
    dataflow.graph.return %ctrl, %carry : none, f32
  }
}
