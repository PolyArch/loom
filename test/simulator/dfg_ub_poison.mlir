// RUN: loom-dfg-sim %s --graph poison_top --output %t.top.json
// RUN: FileCheck %s --check-prefix=POISON < %t.top.json

// POISON-DAG: "workload": "poison_top"
// POISON-DAG: "status": "pass"
// POISON-DAG: "ub.poison": 1
// POISON-DAG: "i32:poison"

module {
  dataflow.graph private @poison_top(%ctrl: none) -> (i32)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %poison = ub.poison : i32
    %published:2 = dataflow.sync %ctrl, %poison
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
