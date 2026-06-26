// RUN: loom-dfg-sim %s --graph pack4 --arg 0=none --arg 1=1 --arg 1=2 --arg 1=3 --arg 1=99 --arg 2=true --arg 2=true --arg 2=true --arg 2=false --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "pack4"
// CHECK-DAG: "graph": "pack4"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.parallelize": 4
// CHECK-DAG: "dataflow.pack": 1
// CHECK-DAG: "final_outputs":
// CHECK-DAG: "i32:197121"
// CHECK-DAG: "i4:7"

module {
  dataflow.graph.func private @pack4(%ctrl: none, %data: i8, %cont: i1)
      -> (none, i32, i4) {
    %lane0, %lane1, %lane2, %lane3, %mask =
      dataflow.parallelize %data, %cont {vec_size = 4 : i64}
        : (i8, i1) -> (i8, i8, i8, i8, i4)
    %packed = dataflow.pack %lane0, %lane1, %lane2, %lane3 mask %mask
        {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> i32
    dataflow.graph.return %ctrl, %packed, %mask : none, i32, i4
  }
}

// RUN: loom-dfg-sim %s --graph pack_overwide --arg 0=none --arg 1=1 --arg 2=2 --arg 3=3 --arg 4=4 --arg 5=15 --output %t.overwide.json
// RUN: FileCheck %s --check-prefix=OVERWIDE < %t.overwide.json

// OVERWIDE-DAG: "workload": "pack_overwide"
// OVERWIDE-DAG: "status": "blocked"
// OVERWIDE-DAG: "final_outputs":
// OVERWIDE-DAG: "missing"
// OVERWIDE-DAG: "dataflow.pack DFG-sim supports packed widths up to 64 bits"

module {
  dataflow.graph.func private @pack_overwide(%ctrl: none, %a: i32, %b: i32,
                                             %c: i32, %d: i32, %mask: i4)
      -> (none, i128) {
    %packed = dataflow.pack %a, %b, %c, %d mask %mask {vec_size = 4 : i64}
      : (i32, i32, i32, i32, i4) -> i128
    dataflow.graph.return %ctrl, %packed : none, i128
  }
}

// RUN: loom-dfg-sim %s --graph pack_partial_unflushed --arg 0=none --arg 1=1 --arg 1=2 --arg 1=3 --arg 1=4 --arg 1=5 --arg 2=true --arg 2=true --arg 2=true --arg 2=true --arg 2=true --output %t.partial.json
// RUN: FileCheck %s --check-prefix=PARTIAL < %t.partial.json

// PARTIAL-DAG: "workload": "pack_partial_unflushed"
// PARTIAL-DAG: "status": "blocked"
// PARTIAL-DAG: "dataflow.parallelize": 5
// PARTIAL-DAG: "dataflow.pack": 1
// PARTIAL-DAG: "dataflow.parallelize ended with pending lanes; emit a false continuation token to flush the partial vector group"

module {
  dataflow.graph.func private @pack_partial_unflushed(%ctrl: none, %data: i8,
                                                      %cont: i1)
      -> (none, i32, i4) {
    %lane0, %lane1, %lane2, %lane3, %mask =
      dataflow.parallelize %data, %cont {vec_size = 4 : i64}
        : (i8, i1) -> (i8, i8, i8, i8, i4)
    %packed = dataflow.pack %lane0, %lane1, %lane2, %lane3 mask %mask
      {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> i32
    dataflow.graph.return %ctrl, %packed, %mask : none, i32, i4
  }
}
