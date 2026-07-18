// RUN: loom-dfg-sim %s --graph pack4 --arg 0=1 --arg 1=2 --arg 2=3 --arg 3=99 --arg 4=7 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "graph": "pack4"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.pack": 1
// CHECK-DAG: "i32:197121"

module {
  dataflow.graph private @pack4(
      %ctrl: none, %lane0: i8, %lane1: i8, %lane2: i8, %lane3: i8,
      %mask: i4) -> i32
      attributes {input_segments = array<i32: 5, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %packed = dataflow.pack %lane0, %lane1, %lane2, %lane3 mask %mask
        {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> i32
    %published:2 = dataflow.sync %ctrl, %packed
        : (none, i32) -> (none, i32)
    dataflow.graph.return %published#0, %published#1 : none, i32
  }
}
