// RUN: loom-pnr-map --dfg-mlir %s --graph rle_decode --hardware-mlir %S/shared_memory_reduction_adg.mlir --hardware shared_memory_reduction_adg --workload rle_decode --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: rle_decode,shared_memory_reduction_adg,rle_decode__rle_decode__shared_memory_reduction_adg,6,4,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "operation": "arith.addi"
// JSON-DAG: "operation": "arith.cmpi"
// JSON-DAG: "operation": "dataflow.load"
// JSON-DAG: "operation": "dataflow.store"
// JSON-NOT: "resource_pressure"

module {
  dataflow.graph.func private @rle_decode(
      %ctrl: none,
      %start: index,
      %end: index,
      %step: index,
      %values: !llvm.ptr,
      %counts: !llvm.ptr,
      %zero: i32,
      %output: !llvm.ptr,
      %write_start: i32) -> (none, i32) {
    %final = scf.for %i = %start to %end step %step iter_args(%write = %write_start) -> (i32) {
      %values_mem = builtin.unrealized_conversion_cast %values : !llvm.ptr to memref<?xi32>
      %value, %value_done = dataflow.load %values_mem[%i] %ctrl : memref<?xi32>
      %counts_mem = builtin.unrealized_conversion_cast %counts : !llvm.ptr to memref<?xi32>
      %count, %count_done = dataflow.load %counts_mem[%i] %ctrl : memref<?xi32>
      %empty = arith.cmpi eq, %count, %zero : i32
      %next_write = scf.if %empty -> (i32) {
        scf.yield %write : i32
      } else {
        %limit_i32 = arith.addi %write, %count : i32
        %write_index = arith.index_cast %write : i32 to index
        %write_index_0 = arith.index_cast %write : i32 to index
        %count_index_0 = arith.index_cast %count : i32 to index
        %limit = arith.addi %write_index_0, %count_index_0 : index
        scf.forall (%j) = (%write_index) to (%limit) step (1) {
          %output_mem = builtin.unrealized_conversion_cast %output : !llvm.ptr to memref<?xi32>
          %stored = dataflow.store %output_mem[%j] %value %ctrl : memref<?xi32>
        }
        scf.yield %limit_i32 : i32
      }
      scf.yield %next_write : i32
    } {loom.stream_cont_cond = "<"}
    dataflow.graph.return %ctrl, %final : none, i32
  }
}
