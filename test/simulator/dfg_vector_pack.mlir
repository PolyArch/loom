// RUN: loom-dfg-sim %s --graph float_bits --arg 0=9205377655040376832 --output %t.float.json
// RUN: FileCheck %s --check-prefix=FLOAT < %t.float.json
// RUN: loom-dfg-sim %s --graph wide_bits --arg 0=78876037339089764361607826927 --output %t.wide.json
// RUN: FileCheck %s --check-prefix=WIDE < %t.wide.json
// RUN: loom-dfg-sim %s --graph vector_add --arg 0=513 --arg 1=1027 --output %t.add.json
// RUN: FileCheck %s --check-prefix=ADD < %t.add.json
// RUN: loom-dfg-sim %s --graph unsupported_vector_add \
// RUN:   --arg 0=1 --arg 1=2 --output %t.unsupported-add.json
// RUN: FileCheck %s --check-prefix=UNSUPPORTED-ADD < %t.unsupported-add.json

// FLOAT-DAG: "graph": "float_bits"
// FLOAT-DAG: "status": "pass"
// FLOAT-DAG: "dataflow.unpack": 1
// FLOAT-DAG: "dataflow.pack": 1
// FLOAT-DAG: "i64:9205377655040376832"

// WIDE-DAG: "graph": "wide_bits"
// WIDE-DAG: "status": "pass"
// WIDE-DAG: "dataflow.unpack": 1
// WIDE-DAG: "dataflow.pack": 1
// WIDE-DAG: "i96:-352125175174573231936123409"

// ADD-DAG: "graph": "vector_add"
// ADD-DAG: "status": "pass"
// ADD-DAG: "arith.addi": 1
// ADD-DAG: "vector<2xi8>:0x604"

// UNSUPPORTED-ADD-DAG: "event_count": 0
// UNSUPPORTED-ADD-DAG: "status": "unsupported"
// UNSUPPORTED-ADD-DAG: "unsupported op: arith.addi"

module {
  dataflow.graph private @float_bits(
      %start: none, %packed: i64) -> i64
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %vector = dataflow.unpack %packed : i64 -> vector<2xf32>
    %roundtrip = dataflow.pack %vector : vector<2xf32> -> i64
    %published:2 = dataflow.sync %start, %roundtrip
        : (none, i64) -> (none, i64)
    dataflow.graph.return %published#0, %published#1 : none, i64
  }

  dataflow.graph private @wide_bits(
      %start: none, %packed: i96) -> i96
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %vector = dataflow.unpack %packed : i96 -> vector<3xi32>
    %roundtrip = dataflow.pack %vector : vector<3xi32> -> i96
    %published:2 = dataflow.sync %start, %roundtrip
        : (none, i96) -> (none, i96)
    dataflow.graph.return %published#0, %published#1 : none, i96
  }

  dataflow.graph private @vector_add(
      %start: none, %packed_lhs: i16, %packed_rhs: i16) -> vector<2xi8>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.unpack %packed_lhs : i16 -> vector<2xi8>
    %rhs = dataflow.unpack %packed_rhs : i16 -> vector<2xi8>
    %sum = arith.addi %lhs, %rhs : vector<2xi8>
    %published:2 = dataflow.sync %start, %sum
        : (none, vector<2xi8>) -> (none, vector<2xi8>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<2xi8>
  }

  dataflow.graph private @unsupported_vector_add(
      %start: none, %packed_lhs: i192, %packed_rhs: i192) -> vector<2xi96>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.unpack %packed_lhs : i192 -> vector<2xi96>
    %rhs = dataflow.unpack %packed_rhs : i192 -> vector<2xi96>
    %sum = arith.addi %lhs, %rhs : vector<2xi96>
    %published:2 = dataflow.sync %start, %sum
        : (none, vector<2xi96>) -> (none, vector<2xi96>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<2xi96>
  }
}
