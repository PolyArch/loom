// RUN: loom-dfg-sim %s --graph float_bits --arg 0=9205377655040376832 --output %t.float.json
// RUN: FileCheck %s --check-prefix=FLOAT < %t.float.json
// RUN: loom-dfg-sim %s --graph wide_bits --arg 0=78876037339089764361607826927 --output %t.wide.json
// RUN: FileCheck %s --check-prefix=WIDE < %t.wide.json

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
}
