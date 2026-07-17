// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/shared-memory-reduction.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph rle_decode --hardware-mlir %t.dir/shared-memory-reduction.mlir --hardware shared_memory_reduction_adg --workload rle_decode --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: rle_decode,shared_memory_reduction_adg,rle_decode__rle_decode__shared_memory_reduction_adg,10,15,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "operation": "arith.addi"
// JSON-DAG: "operation": "arith.cmpi"
// JSON-DAG: "operation": "arith.select"
// JSON-DAG: "operation": "dataflow.load"
// JSON-DAG: "operation": "dataflow.store"
// JSON-DAG: "operation": "dataflow.mux"
// JSON-NOT: "resource_pressure"

module {
  dataflow.graph private @rle_decode(
      %ctrl: none,
      %index: index,
      %write: i32,
      %zero: i32,
      %values: memref<?xi32>,
      %counts: memref<?xi32>,
      %output: memref<?xi32>) -> (i32)
      attributes {input_segments = array<i32: 3, 0, 3>,
                  result_segments = array<i32: 1, 0, 0>} {
    %value, %value_done = dataflow.load %values[%index] %ctrl
        : memref<?xi32>
    %count, %count_done = dataflow.load %counts[%index] %ctrl
        : memref<?xi32>
    %empty = arith.cmpi eq, %count, %zero : i32
    %limit = arith.addi %write, %count : i32
    %next_write = arith.select %empty, %write, %limit : i32
    %write_index = arith.index_cast %write : i32 to index
    %loaded:2 = dataflow.sync %value_done, %count_done
        : (none, none) -> (none, none)
    %paths:2 = dataflow.demux %empty, %loaded#0
        : (i1, none) -> (none, none)
    %stored = dataflow.store %output[%write_index] %value %paths#0
        : memref<?xi32>
    %retired = dataflow.mux %empty, %stored, %paths#1
        : (i1, none, none) -> none
    %published:2 = dataflow.sync %retired, %next_write
        : (none, i32) -> (none, i32)
    dataflow.graph.return values(%published#1 : i32) streams() memories()
        complete(%published#0 : none)
  }
}
