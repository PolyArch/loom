// RUN: loom-dfg-sim %s --graph unpack_serialize --arg 0=197121 --arg 1=7 --output %t.json
// RUN: FileCheck %s < %t.json

// CHECK-DAG: "kind": "dfg_sim_report"
// CHECK-DAG: "workload": "unpack_serialize"
// CHECK-DAG: "graph": "unpack_serialize"
// CHECK-DAG: "status": "pass"
// CHECK-DAG: "dataflow.unpack": 1
// CHECK-DAG: "dataflow.serialize": 1
// CHECK-DAG: "final_stream_outputs":
// CHECK-DAG: "i8:3"
// CHECK-DAG: "i1:false"

module {
  dataflow.graph private @unpack_serialize(%ctrl: none, %packed: i32,
                                                %mask: i4) -> (i8, i1)
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

// RUN: loom-dfg-sim %s --graph unpack_signless_bits --arg 0=65535 --arg 1=3 --output %t.signless.json
// RUN: FileCheck %s --check-prefix=SIGNLESS < %t.signless.json

// SIGNLESS-DAG: "workload": "unpack_signless_bits"
// SIGNLESS-DAG: "status": "pass"
// SIGNLESS-DAG: "dataflow.unpack": 1
// SIGNLESS-DAG: "final_outputs":
// SIGNLESS-DAG: "i8:255"
// SIGNLESS-DAG: "i8:255"

module {
  dataflow.graph private @unpack_signless_bits(%ctrl: none,
                                                    %packed: i16,
                                                    %mask: i2)
      -> (i8, i8)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %lane0, %lane1 = dataflow.unpack %packed, %mask {vec_size = 2 : i64}
      : (i16, i2) -> (i8, i8)
    %published:3 = dataflow.sync %ctrl, %lane0, %lane1
        : (none, i8, i8) -> (none, i8, i8)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, i8, i8
  }
}

// RUN: loom-dfg-sim %s --graph parallelize_bad_stride --arg 0=7 --arg 1=0 --output %t.stride.json
// RUN: FileCheck %s --check-prefix=BAD-STRIDE < %t.stride.json

// BAD-STRIDE-DAG: "workload": "parallelize_bad_stride"
// BAD-STRIDE-DAG: "status": "blocked"
// BAD-STRIDE-DAG: "final_outputs":
// BAD-STRIDE-DAG: "missing"
// BAD-STRIDE-DAG: "dataflow.parallelize stride must be positive"

module {
  dataflow.graph private @parallelize_bad_stride(%ctrl: none,
                                                      %data: i8,
                                                      %stride: i8)
      -> (i8, i4)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 2, 0, 0>} {
    %c0 = dataflow.constant %ctrl {const_value = 0 : i8} : i8
    %c1 = dataflow.constant %ctrl {const_value = 1 : i8} : i8
    %iv, %cont = dataflow.stream %c0, %c1, %c1
        step add while slt : i8
    %data_stream = dataflow.invariant %cont, %data : i8
    %stride_stream = dataflow.invariant %cont, %stride : i8
    %lane0, %lane1, %lane2, %lane3, %mask =
      dataflow.parallelize %data_stream, %cont, %stride_stream
        {vec_size = 4 : i64}
        : (i8, i1, i8) -> (i8, i8, i8, i8, i4)
    %tokens = dataflow.invariant %cont, %ctrl : none
    %close:2 = dataflow.demux %cont, %tokens
        : (i1, none) -> (none, none)
    %published:3 = dataflow.sync %close#0, %lane0, %mask
        : (none, i8, i4) -> (none, i8, i4)
    dataflow.graph.return %published#0, %published#1, %published#2
        : none, i8, i4
  }
}

// RUN: loom-dfg-sim %s --graph serialize_zero_mask_returned_data --arg 0=0 --arg 1=0 --output %t.zero-mask.json
// RUN: FileCheck %s --check-prefix=ZERO-MASK < %t.zero-mask.json

// ZERO-MASK-DAG: "workload": "serialize_zero_mask_returned_data"
// ZERO-MASK-DAG: "status": "pass"
// ZERO-MASK-DAG: "dataflow.unpack": 1
// ZERO-MASK-DAG: "dataflow.serialize": 1
// ZERO-MASK-DAG: "final_stream_outputs":
// ZERO-MASK-DAG: []
// ZERO-MASK-DAG: "i1:false"

module {
  dataflow.graph private @serialize_zero_mask_returned_data(%ctrl: none,
                                                                 %packed: i32,
                                                                 %mask: i4)
      -> (i8, i1)
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
