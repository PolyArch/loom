// RUN: loom-dfg-sim %s --graph unpack_serialize --arg 0=197121 --arg 1=7 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "graph": "unpack_serialize"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.unpack": 1
// CHECK-DAG: "dataflow.serialize": 1
// CHECK-DAG: "final_stream_outputs":
// CHECK-DAG: "i8:3"
// CHECK-DAG: "i1:false"

module {
  dataflow.graph private @unpack_serialize(
      %ctrl: none, %packed: i32, %mask: i4) -> (i8, i1)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 0, 2, 0>} {
    %lane0, %lane1, %lane2, %lane3 =
      dataflow.unpack %packed, %mask {vec_size = 4 : i64}
        : (i32, i4) -> (i8, i8, i8, i8)
    %data, %cont =
      dataflow.serialize %lane0, %lane1, %lane2, %lane3 mask %mask
        {vec_size = 4 : i64} : (i8, i8, i8, i8, i4) -> (i8, i1)
    %tokens = dataflow.invariant %cont, %ctrl : none
    %complete:2 = dataflow.demux %cont, %tokens
        : (i1, none) -> (none, none)
    dataflow.graph.return values() streams(%data, %cont : i8, i1) memories()
        complete(%complete#0 : none)
  }
}
