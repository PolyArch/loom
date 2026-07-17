// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-dfg-sim %t.dir/none-witness.mlir --graph none_witnesses_value --arg 0=7 --output %t.none.json
// RUN: FileCheck %s --check-prefix=NONE < %t.none.json
// RUN: loom-dfg-sim %t.dir/value-witness.mlir --graph value_witnesses_none --arg 0=9 --output %t.value.json
// RUN: FileCheck %s --check-prefix=VALUE < %t.value.json
// RUN: not loom-dfg-sim %t.dir/unrelated.mlir --graph unrelated_load_siblings --memref 0=11 --output %t.unrelated.json 2>&1 | FileCheck %s --check-prefix=UNRELATED

// NONE-DAG: "status": "pass"
// NONE-DAG: "dataflow.sync": 1
// NONE-DAG: "i32:7"

// VALUE-DAG: "status": "pass"
// VALUE-DAG: "dataflow.sync": 2
// VALUE-DAG: "final_stream_outputs": [
// VALUE-DAG: "none"
// VALUE-DAG: "i32:9"

// UNRELATED: retirement frontier does not causally cover stream output #0

// A none result covers its value sibling because both are published by one
// dataflow.sync firing.
//--- none-witness.mlir
module {
  dataflow.graph private @none_witnesses_value(
      %start: none, %value: i32) -> (i32)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %published:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
}

// Retirement flows through the value result before covering the none sibling
// from the same atomic publication group.
//--- value-witness.mlir
module {
  dataflow.graph private @value_witnesses_none(
      %start: none, %value: i32) -> (i32, none)
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 1, 1, 0>} {
    %published:2 = dataflow.sync %start, %value
        : (none, i32) -> (none, i32)
    %retired:2 = dataflow.sync %published#1, %start
        : (i32, none) -> (i32, none)
    dataflow.graph.return values(%retired#0 : i32)
        streams(%published#0 : none) memories()
        complete(%retired#1 : none)
  }
}

// Multi-result operations other than dataflow.sync do not gain sibling
// completion coverage.
//--- unrelated.mlir
module {
  dataflow.graph private @unrelated_load_siblings(
      %start: none, %memory: memref<?xi32>) -> (i32, none)
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 1, 1, 0>} {
    %index = dataflow.constant %start {const_value = 0 : index} : index
    %data, %loaded = dataflow.load %memory[%index] %start : memref<?xi32>
    %retired:2 = dataflow.sync %data, %start
        : (i32, none) -> (i32, none)
    dataflow.graph.return values(%retired#0 : i32)
        streams(%loaded : none) memories() complete(%retired#1 : none)
  }
}
