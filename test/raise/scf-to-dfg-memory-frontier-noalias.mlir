// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s
// RUN: loom %s | loom-raise-opt --loom-lower-graph-memory | FileCheck %s
// RUN: loom %s --emit-bytecode | loom-raise-opt --loom-lower-graph-memory | FileCheck %s

// Explicit noalias metadata keeps boundary memories in distinct frontier
// partitions after every supported serialization path.
// CHECK-LABEL: dataflow.graph private @frontier_boundary_args_noalias
// CHECK: %[[WRITE:.*]] = dataflow.store %arg3[%arg1] %arg2 %arg0 : memref<?xi32>
// CHECK: %{{.*}}, %[[READ_DONE:.*]] = dataflow.load %arg4[%arg1] %arg0 : memref<?xi32>
dataflow.graph private @frontier_boundary_args_noalias(
    %start: none, %index: index, %value: i32,
    %a: memref<?xi32> {llvm.noalias},
    %b: memref<?xi32> {llvm.noalias}) -> ()
    attributes {input_segments = array<i32: 2, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  memref.store %value, %a[%index] : memref<?xi32>
  %loaded = memref.load %b[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}
