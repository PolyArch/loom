// RUN: loom-dfg-sim %s --graph gate_stream --arg 0=none --arg 1=0 --arg 2=2 --arg 3=1 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "gate_stream"
// CHECK-DAG: "graph": "gate_stream"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "operation_cost_score": 8
// CHECK-DAG: "wavefront_steps": 4
// CHECK-DAG: "event_count": 6
// CHECK-DAG: "dynamic_work_items": 2
// CHECK-DAG: "dataflow.stream": 3
// CHECK-DAG: "dataflow.gate": 3
// CHECK-DAG: "none",
// CHECK-DAG: "i1:false"
// CHECK-DAG: "i64:1"

module {
  dataflow.graph.func private @gate_stream(%ctrl: none, %lb: i64, %ub: i64,
                                           %step: i64) -> (none, i1, i64) {
    %idx, %rwc = dataflow.stream %lb, %ub, %step {step_op = "+=", cont_cond = "<"} : i64
    %after_cond, %after_value = dataflow.gate %rwc, %idx : i64
    dataflow.graph.return %ctrl, %after_cond, %after_value : none, i1, i64
  }
}
