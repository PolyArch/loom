// RUN: loom-dfg-sim %s --graph float_bits --arg 0=9205377655040376832 --output %t.float.json
// RUN: FileCheck %s --check-prefix=FLOAT < %t.float.json
// RUN: loom-dfg-sim %s --graph wide_bits --arg 0=78876037339089764361607826927 --output %t.wide.json
// RUN: FileCheck %s --check-prefix=WIDE < %t.wide.json
// RUN: loom-dfg-sim %s --graph f80_bits --arg 0=1 --output %t.f80.json
// RUN: FileCheck %s --check-prefix=F80 < %t.f80.json
// RUN: loom-dfg-sim %s --graph rank_two_bits --arg 0=6618611909121 --output %t.rank2.json
// RUN: FileCheck %s --check-prefix=RANK2 < %t.rank2.json
// RUN: loom-dfg-sim %s --graph vector_add --arg 0=513 --arg 1=1027 --output %t.add.json
// RUN: FileCheck %s --check-prefix=ADD < %t.add.json
// RUN: loom-dfg-sim %s --graph unsupported_vector_add \
// RUN:   --arg 0=1 --arg 1=2 --output %t.unsupported-add.json
// RUN: FileCheck %s --check-prefix=UNSUPPORTED-ADD < %t.unsupported-add.json
// RUN: loom-dfg-sim %s --graph unsupported_f80_add \
// RUN:   --arg 0=0 --arg 1=0 --output %t.unsupported-f80.json
// RUN: FileCheck %s --check-prefix=UNSUPPORTED-F80 < %t.unsupported-f80.json

// FLOAT-DAG: "graph": "float_bits"
// FLOAT-DAG: "status": "pass"
// FLOAT-DAG: "dataflow.unpack": 1
// FLOAT-DAG: "dataflow.pack": 1
// FLOAT-DAG: "i64:9205377655040376832"

// WIDE-DAG: "graph": "wide_bits"
// WIDE-DAG: "status": "pass"
// WIDE-DAG: "dataflow.unpack": 1
// WIDE-DAG: "dataflow.pack": 1
// WIDE-DAG: "i96:78876037339089764361607826927"

// F80-DAG: "graph": "f80_bits"
// F80-DAG: "status": "pass"
// F80-DAG: "dataflow.unpack": 1
// F80-DAG: "dataflow.pack": 1
// F80-DAG: "i80:1"

// RANK2-DAG: "graph": "rank_two_bits"
// RANK2-DAG: "status": "pass"
// RANK2-DAG: "dataflow.unpack": 1
// RANK2-DAG: "dataflow.pack": 1
// RANK2-DAG: "i48:6618611909121"

// ADD-DAG: "graph": "vector_add"
// ADD-DAG: "status": "pass"
// ADD-DAG: "arith.addi": 1
// ADD-DAG: "vector<2xi8>:0x604"

// UNSUPPORTED-ADD-DAG: "event_count": 0
// UNSUPPORTED-ADD-DAG: "status": "unsupported"
// UNSUPPORTED-ADD-DAG: "unsupported op: arith.addi: result element type i96 has width 96; scalar primitive evaluator supports integer lane widths from 1 to 64"

// UNSUPPORTED-F80-DAG: "event_count": 0
// UNSUPPORTED-F80-DAG: "status": "unsupported"
// UNSUPPORTED-F80-DAG: "unsupported op: arith.addf: result element type f80 has 80-bit floating-point semantics not exactly representable by the scalar evaluator's f64 lane model"

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

  dataflow.graph private @f80_bits(
      %start: none, %packed: i80) -> i80
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %vector = dataflow.unpack %packed : i80 -> vector<1xf80>
    %roundtrip = dataflow.pack %vector : vector<1xf80> -> i80
    %published:2 = dataflow.sync %start, %roundtrip
        : (none, i80) -> (none, i80)
    dataflow.graph.return %published#0, %published#1 : none, i80
  }

  dataflow.graph private @rank_two_bits(
      %start: none, %packed: i48) -> i48
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %vector = dataflow.unpack %packed : i48 -> vector<2x3xi8>
    %roundtrip = dataflow.pack %vector : vector<2x3xi8> -> i48
    %published:2 = dataflow.sync %start, %roundtrip
        : (none, i48) -> (none, i48)
    dataflow.graph.return %published#0, %published#1 : none, i48
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

  dataflow.graph private @unsupported_f80_add(
      %start: none, %packed_lhs: i80, %packed_rhs: i80) -> vector<1xf80>
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %lhs = dataflow.unpack %packed_lhs : i80 -> vector<1xf80>
    %rhs = dataflow.unpack %packed_rhs : i80 -> vector<1xf80>
    %sum = arith.addf %lhs, %rhs : vector<1xf80>
    %published:2 = dataflow.sync %start, %sum
        : (none, vector<1xf80>) -> (none, vector<1xf80>)
    dataflow.graph.return %published#0, %published#1
        : none, vector<1xf80>
  }
}
