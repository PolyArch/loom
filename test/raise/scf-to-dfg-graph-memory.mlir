// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// Positive case: a graph.func body with the canonical
// (dataflow.stream + dataflow.carry + llvm.gep + llvm.load + arith +
// llvm.store) shape gets the residual memory ops tokenized into
// dataflow.load / dataflow.store. The graph block-arg !llvm.ptr is
// bridged to memref<?xf32> via builtin.unrealized_conversion_cast and
// the gep's i64 index becomes the index-typed address port through
// arith.index_cast.

// CHECK-LABEL: dataflow.graph.func private @g_canonical
// CHECK-DAG: %[[IDX:.*]] = arith.index_cast %index : i64 to index
// CHECK-DAG: %[[MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream
// CHECK: dataflow.load %[[MEM]][%[[IDX]]] %arg0 : memref<?xf32>
// CHECK: dataflow.store %[[MEM]][%[[IDX]]] %{{.*}} %arg0 : memref<?xf32>
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_canonical(%arg0: none, %arg1: i64, %arg2: i64,
                                         %arg3: i64, %arg4: !llvm.ptr,
                                         %arg5: f32) -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.carry %rwc, %arg5, %3 : f32
  %1 = llvm.getelementptr %arg4[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %2 = llvm.load %1 : !llvm.ptr -> f32
  %3 = arith.addf %0, %2 : f32
  llvm.store %3, %1 : f32, !llvm.ptr
  dataflow.graph.return %arg0, %0 : none, f32
}

// Negative-bail #1: a graph.func body whose llvm.load / llvm.store
// use a base pointer derived from a global address-of (not a graph
// block-arg) keeps the original llvm.{load, store, gep} chain.

llvm.mlir.global private @global_buf(dense<0.0> : tensor<8xf32>) : !llvm.array<8 x f32>

// CHECK-LABEL: dataflow.graph.func private @g_global_base
// CHECK: %[[GBL:.*]] = llvm.mlir.addressof @global_buf
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[GBL]][%index]
// CHECK: llvm.load %[[GEP]]
// CHECK-NOT: dataflow.load
// CHECK-NOT: dataflow.store
// CHECK: llvm.store %{{.*}}, %[[GEP]]
dataflow.graph.func private @g_global_base(%arg0: none, %arg1: i64, %arg2: i64,
                                           %arg3: i64, %arg5: f32)
    -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.carry %rwc, %arg5, %3 : f32
  %p = llvm.mlir.addressof @global_buf : !llvm.ptr
  %1 = llvm.getelementptr %p[%index] : (!llvm.ptr, i64) -> !llvm.ptr, f32
  %2 = llvm.load %1 : !llvm.ptr -> f32
  %3 = arith.addf %0, %2 : f32
  llvm.store %3, %1 : f32, !llvm.ptr
  dataflow.graph.return %arg0, %0 : none, f32
}

// Byte-offset GEPs from LLVM i8 pointer arithmetic must be normalized to the
// element index used by dataflow.load/store memrefs.

// CHECK-LABEL: dataflow.graph.func private @g_i8_byte_offset_f32
// CHECK-DAG: %[[BYTE:.*]] = arith.shli %index, %arg5 : i64
// CHECK-DAG: %[[ELEM0:.*]] = arith.shrui %[[BYTE]], %{{.*}} : i64
// CHECK-DAG: %[[IDX0:.*]] = arith.index_cast %[[ELEM0]] : i64 to index
// CHECK-DAG: %[[ELEM1:.*]] = arith.shrui %[[BYTE]], %{{.*}} : i64
// CHECK-DAG: %[[IDX1:.*]] = arith.index_cast %[[ELEM1]] : i64 to index
// CHECK-DAG: %[[MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK: dataflow.load %[[MEM]][%[[IDX1]]] %arg0 : memref<?xf32>
// CHECK: dataflow.store %[[MEM]][%[[IDX0]]] %{{.*}} %arg0 : memref<?xf32>
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_i8_byte_offset_f32(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr,
    %arg5: i64, %arg6: f32) -> (none, f32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.carry %rwc, %arg6, %4 : f32
  %1 = arith.shli %index, %arg5 : i64
  %2 = llvm.getelementptr %arg4[%1] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %3 = llvm.load %2 : !llvm.ptr -> f32
  %4 = arith.addf %0, %3 : f32
  llvm.store %4, %2 : f32, !llvm.ptr
  dataflow.graph.return %arg0, %0 : none, f32
}
