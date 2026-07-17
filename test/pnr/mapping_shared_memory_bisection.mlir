// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: loom-adg-builder-test --shared-memory-reduction --output %t.dir/shared-memory-reduction.mlir
// RUN: loom-pnr-map --dfg-mlir %s --graph bisection_step --hardware-mlir %t.dir/shared-memory-reduction.mlir --hardware shared_memory_reduction_adg --workload bisection_step --output %t.mapping.csv --artifact %t.mapping.json
// RUN: FileCheck %s --check-prefix=CSV < %t.mapping.csv
// RUN: FileCheck %s --check-prefix=JSON < %t.mapping.json

// CSV: bisection_step,shared_memory_reduction_adg,bisection_step__bisection_step__shared_memory_reduction_adg,13,20,0,0,pass

// JSON-DAG: "status": "pass"
// JSON-DAG: "hardware": "shared_memory_reduction_adg"
// JSON-DAG: "operation": "arith.addf"
// JSON-DAG: "operation": "arith.mulf"
// JSON-DAG: "operation": "arith.cmpf"
// JSON-DAG: "operation": "arith.select"
// JSON-DAG: "operation": "dataflow.sync"
// JSON-DAG: "operation": "dataflow.load"
// JSON-DAG: "operation": "dataflow.store"
// JSON-NOT: "resource_pressure"

module {
  dataflow.graph.func private @bisection_step(
      %ctrl: none,
      %half: f32,
      %zero: f32,
      %index: index,
      %input_a: !llvm.ptr,
      %input_b: !llvm.ptr,
      %input_fa: !llvm.ptr,
      %input_fc: !llvm.ptr,
      %output_a: !llvm.ptr,
      %output_b: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 3, 0, 6>,
                  result_segments = array<i32: 0, 0, 0>} {
    %out_b = builtin.unrealized_conversion_cast %output_b : !llvm.ptr to memref<?xf32>
    %out_a = builtin.unrealized_conversion_cast %output_a : !llvm.ptr to memref<?xf32>
    %fc = builtin.unrealized_conversion_cast %input_fc : !llvm.ptr to memref<?xf32>
    %fa = builtin.unrealized_conversion_cast %input_fa : !llvm.ptr to memref<?xf32>
    %b = builtin.unrealized_conversion_cast %input_b : !llvm.ptr to memref<?xf32>
    %a = builtin.unrealized_conversion_cast %input_a : !llvm.ptr to memref<?xf32>
    %data_a, %done_a = dataflow.load %a[%index] %ctrl : memref<?xf32>
    %data_b, %done_b = dataflow.load %b[%index] %ctrl : memref<?xf32>
    %sum = arith.addf %data_a, %data_b : f32
    %mid = arith.mulf %sum, %half : f32
    %data_fa, %done_fa = dataflow.load %fa[%index] %ctrl : memref<?xf32>
    %data_fc, %done_fc = dataflow.load %fc[%index] %ctrl : memref<?xf32>
    %product = arith.mulf %data_fa, %data_fc : f32
    %same_side = arith.cmpf olt, %product, %zero : f32
    %next_a = arith.select %same_side, %data_a, %mid : f32
    %next_b = arith.select %same_side, %mid, %data_b : f32
    %store_a = dataflow.store %out_a[%index] %next_a %ctrl : memref<?xf32>
    %store_b = dataflow.store %out_b[%index] %next_b %ctrl : memref<?xf32>
    %sync:6 = dataflow.sync %done_a, %done_b, %done_fa, %done_fc, %store_a, %store_b
        : (none, none, none, none, none, none) -> (none, none, none, none, none, none)
    dataflow.graph.return %sync#0 : none
  }
}
