// RUN: loom-dfg-sim %s --graph pack4 --output %t.json
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
  dataflow.graph private @pack4(%ctrl: none) -> (i32, i4)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %c0 = dataflow.constant %ctrl {const_value = 0 : i8} : i8
    %c1 = dataflow.constant %ctrl {const_value = 1 : i8} : i8
    %c3 = dataflow.constant %ctrl {const_value = 3 : i8} : i8
    %c4 = dataflow.constant %ctrl {const_value = 4 : i8} : i8
    %c99 = dataflow.constant %ctrl {const_value = 99 : i8} : i8
    %iv, %outer_phase = dataflow.stream %c0, %c4, %c1
        step add while slt : i8
    %one = dataflow.invariant %outer_phase, %c1 : i8
    %three = dataflow.invariant %outer_phase, %c3 : i8
    %ninety_nine = dataflow.invariant %outer_phase, %c99 : i8
    %incremented = arith.addi %iv, %one : i8
    %last = arith.cmpi eq, %iv, %three : i8
    %data = arith.select %last, %ninety_nine, %incremented : i8
    %cont = arith.cmpi ne, %iv, %three : i8
    %lane0, %lane1, %lane2, %lane3, %mask =
      dataflow.parallelize %data, %cont {vec_size = 4 : i64}
        : (i8, i1) -> (i8, i8, i8, i8, i4)
    %packed = dataflow.pack %lane0, %lane1, %lane2, %lane3 mask %mask
        {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> i32
    %outer_tokens = dataflow.invariant %outer_phase, %ctrl : none
    %outer_close:2 = dataflow.demux %outer_phase, %outer_tokens
        : (i1, none) -> (none, none)
    %parallel_tokens = dataflow.invariant %cont, %ctrl : none
    %parallel_close:2 = dataflow.demux %cont, %parallel_tokens
        : (i1, none) -> (none, none)
    %closed:2 = dataflow.sync %outer_close#0, %parallel_close#0
        : (none, none) -> (none, none)
    %published:3 = dataflow.sync %closed#0, %packed, %mask
        : (none, i32, i4) -> (none, i32, i4)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, i32, i4
  }
}

// RUN: loom-dfg-sim %s --graph pack_overwide --arg 0=1 --arg 1=2 --arg 2=3 --arg 3=4 --arg 4=15 --output %t.overwide.json
// RUN: FileCheck %s --check-prefix=OVERWIDE < %t.overwide.json

// OVERWIDE-DAG: "workload": "pack_overwide"
// OVERWIDE-DAG: "status": "blocked"
// OVERWIDE-DAG: "final_outputs":
// OVERWIDE-DAG: "missing"
// OVERWIDE-DAG: "dataflow.pack DFG-sim supports packed widths up to 64 bits"

module {
  dataflow.graph private @pack_overwide(%ctrl: none, %a: i32, %b: i32,
                                             %c: i32, %d: i32, %mask: i4)
      -> (i128)
      attributes {input_segments = array<i32: 5, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %packed = dataflow.pack %a, %b, %c, %d mask %mask {vec_size = 4 : i64}
      : (i32, i32, i32, i32, i4) -> i128
    %published:2 = dataflow.sync %ctrl, %packed
        : (none, i128) -> (none, i128)
    dataflow.graph.return %published#0, %published#1 : none, i128
  }
}

// RUN: loom-dfg-sim %s --graph pack_partial_unflushed --output %t.partial.json
// RUN: FileCheck %s --check-prefix=PARTIAL < %t.partial.json

// PARTIAL-DAG: "workload": "pack_partial_unflushed"
// PARTIAL-DAG: "status": "blocked"
// PARTIAL-DAG: "dataflow.parallelize": 5
// PARTIAL-DAG: "dataflow.pack": 1
// PARTIAL-DAG: "dataflow.parallelize ended with pending lanes; emit a false continuation token to flush the partial vector group"

module {
  dataflow.graph private @pack_partial_unflushed(%ctrl: none)
      -> (i32, i4)
      attributes {input_segments = array<i32: 0, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %c0 = dataflow.constant %ctrl {const_value = 0 : i8} : i8
    %c1 = dataflow.constant %ctrl {const_value = 1 : i8} : i8
    %c5 = dataflow.constant %ctrl {const_value = 5 : i8} : i8
    %data, %cont = dataflow.stream %c0, %c5, %c1
        step add while slt : i8
    %lane0, %lane1, %lane2, %lane3, %mask =
      dataflow.parallelize %data, %cont {vec_size = 4 : i64}
        : (i8, i1) -> (i8, i8, i8, i8, i4)
    %packed = dataflow.pack %lane0, %lane1, %lane2, %lane3 mask %mask
      {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> i32
    %tokens = dataflow.invariant %cont, %ctrl : none
    %close:2 = dataflow.demux %cont, %tokens
        : (i1, none) -> (none, none)
    %published:3 = dataflow.sync %close#0, %packed, %mask
        : (none, i32, i4) -> (none, i32, i4)
    %always_true = dataflow.constant %ctrl {const_value = true} : i1
    %never:2 = dataflow.demux %always_true, %ctrl
        : (i1, none) -> (none, none)
    dataflow.graph.return values(%published#1, %published#2 : i32, i4)
        streams() memories() complete(%published#0, %never#0 : none, none)
  }
}
