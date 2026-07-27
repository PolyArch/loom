// RUN: loom-dfg-sim %s --graph zext --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "zext"
// CHECK-DAG: "graph": "zext"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "wavefront_steps": 3
// CHECK-DAG: "event_count": 3
// CHECK-DAG: "i64:42"

module {
  dataflow.graph private @zext(%ctrl: none) -> (i64)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %narrow = dataflow.constant %ctrl {const_value = 42 : i32} : i32
    %wide = arith.extui %narrow : i32 to i64
    %published:2 = dataflow.sync %ctrl, %wide
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }
}
