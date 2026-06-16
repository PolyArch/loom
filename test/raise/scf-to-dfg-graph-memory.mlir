// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// Positive case: a graph.func body with the canonical
// (dataflow.stream + dataflow.carry + llvm.gep + llvm.load + arith +
// llvm.store) shape gets the residual memory ops tokenized into
// dataflow.load / dataflow.store. The graph block-arg !llvm.ptr is
// bridged to memref<?xf32> via builtin.unrealized_conversion_cast and
// the gep's i64 index becomes the index-typed address port through
// arith.index_cast.

// CHECK-LABEL: dataflow.graph.func private @g_canonical
// CHECK-DAG: %[[MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream
// CHECK: %[[IDX:.*]] = arith.index_cast %[[STREAM]] : i64 to index
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
// CHECK-DAG: %[[MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK: %[[ELEM0:.*]] = arith.shrui %[[BYTE]], %{{.*}} : i64
// CHECK: %[[IDX0:.*]] = arith.index_cast %[[ELEM0]] : i64 to index
// CHECK: dataflow.load %[[MEM]][%[[IDX0]]] %arg0 : memref<?xf32>
// CHECK: %[[ELEM1:.*]] = arith.shrui %[[BYTE]], %{{.*}} : i64
// CHECK: %[[IDX1:.*]] = arith.index_cast %[[ELEM1]] : i64 to index
// CHECK: dataflow.store %[[MEM]][%[[IDX1]]] %{{.*}} %arg0 : memref<?xf32>
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

// Pointer induction through a carried LLVM pointer is a memory-view
// concern, not a fabric pointer operation. When the carried pointer advances
// by exactly one element per stream item, memory lowering must bind the
// memref to the original graph pointer and drive load/store addresses from a
// zero-based ordinal counter. The loop IV stream may have nonzero lower bounds
// or non-unit steps, so the raw stream index is not the memory-view offset.
// The residual pointer bookkeeping may remain for graph results, but it must
// no longer be the memory address path.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_i8_f32
// CHECK-DAG: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK-DAG: %[[DST:.*]] = builtin.unrealized_conversion_cast %arg5 : !llvm.ptr to memref<?xf32>
// CHECK: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream
// CHECK: %[[ZERO:.*]] = dataflow.constant %arg0 {const_value = 0 : i32} : i32
// CHECK: %[[ONE:.*]] = dataflow.constant %arg0 {const_value = 1 : i32} : i32
// CHECK: %[[STABLE_ONE:.*]] = dataflow.invariant %[[RWC]], %[[ONE]] : i32
// CHECK: %[[ORD:.*]] = dataflow.carry %[[RWC]], %[[ZERO]], %[[NEXT:.*]] : i32
// CHECK: %[[NEXT]] = arith.addi %[[ORD]], %[[STABLE_ONE]] : i32
// CHECK: %[[IDX:.*]] = arith.index_cast %[[ORD]] : i32 to index
// CHECK-NOT: arith.index_cast %[[STREAM]] : i32 to index
// CHECK: dataflow.load %[[SRC]][%[[IDX]]] %arg0 : memref<?xf32>
// CHECK: dataflow.store %[[DST]][%[[IDX]]] %{{.*}} %arg0 : memref<?xf32>
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_pointer_carry_i8_f32(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: !llvm.ptr,
    %arg5: !llvm.ptr, %bias: f32) -> (none, !llvm.ptr, !llvm.ptr) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i32
  %src_cur = dataflow.carry %rwc, %arg4, %src_next : !llvm.ptr
  %dst_cur = dataflow.carry %rwc, %arg5, %dst_next : !llvm.ptr
  %src_next = llvm.getelementptr %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %data = llvm.load %src_cur : !llvm.ptr -> f32
  %sum = arith.addf %data, %bias : f32
  %dst_next = llvm.getelementptr %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %sum, %dst_cur : f32, !llvm.ptr
  dataflow.graph.return %arg0, %src_cur, %dst_cur : none, !llvm.ptr, !llvm.ptr
}
