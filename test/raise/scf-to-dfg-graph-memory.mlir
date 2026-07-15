// RUN: loom-raise-opt --split-input-file --loom-lower-graph-memory %s | FileCheck %s

// Positive case: a graph.func body with the canonical
// (dataflow.stream + dataflow.carry + llvm.gep + llvm.load + arith +
// llvm.store) shape gets the residual memory ops tokenized into
// dataflow.load / dataflow.store. The graph block-arg !llvm.ptr is
// bridged to memref<?xf32> via builtin.unrealized_conversion_cast. Address
// arithmetic remains in the LLVM pointer-index width until the exact byte
// offset is converted to an element index.

// CHECK-LABEL: dataflow.graph.func private @g_canonical
// CHECK-DAG: %[[MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream
// CHECK: %[[LOAD_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[LOAD_BYTE:.*]] = arith.muli %[[STREAM]], %[[LOAD_STRIDE]] : i64
// CHECK: %[[LOAD_ELEM:.*]] = arith.shrsi %[[LOAD_BYTE]], %{{.*}} : i64
// CHECK: %[[LOAD_IDX:.*]] = arith.index_cast %[[LOAD_ELEM]] : i64 to index
// CHECK: dataflow.load %[[MEM]][%[[LOAD_IDX]]] %arg0 : memref<?xf32>
// CHECK: %[[STORE_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[STORE_BYTE:.*]] = arith.muli %[[STREAM]], %[[STORE_STRIDE]] : i64
// CHECK: %[[STORE_ELEM:.*]] = arith.shrsi %[[STORE_BYTE]], %{{.*}} : i64
// CHECK: %[[STORE_IDX:.*]] = arith.index_cast %[[STORE_ELEM]] : i64 to index
// CHECK: dataflow.store %[[MEM]][%[[STORE_IDX]]] %{{.*}} %arg0 : memref<?xf32>
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

// An arbitrary dynamic i8 byte offset is not known to be divisible by the f32
// element size, so the pass must preserve the LLVM memory operations.

// CHECK-LABEL: dataflow.graph.func private @g_i8_byte_offset_f32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[BYTE:.*]] = arith.shli %index, %arg5 : i64
// CHECK: %[[BYTE_PTR:.*]] = llvm.getelementptr %arg4[%[[BYTE]]] : (!llvm.ptr, i64) -> !llvm.ptr, i8
// CHECK: llvm.load %[[BYTE_PTR]] : !llvm.ptr -> f32
// CHECK: llvm.store %{{.*}}, %[[BYTE_PTR]] : f32, !llvm.ptr
// CHECK-NOT: dataflow.load
// CHECK-NOT: dataflow.store
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

// A dynamic aggregate-stride GEP followed by a constant byte GEP can be
// normalized when both byte contributions convert exactly to load elements.

// CHECK-LABEL: dataflow.graph.func private @g_chained_gep_i8_i16
// CHECK-DAG: %[[MEM_CHAIN:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xi16>
// CHECK: %[[STREAM_CHAIN:.*]], %[[RWC_CHAIN:.*]] = dataflow.stream
// CHECK: %[[STRIDE_CHAIN:.*]] = arith.constant 4 : i64
// CHECK: %[[BASE_BYTES_CHAIN:.*]] = arith.muli %[[STREAM_CHAIN]], %[[STRIDE_CHAIN]] : i64
// CHECK: %[[BIAS_CHAIN:.*]] = arith.constant 2 : i64
// CHECK: %[[ADDR_BYTES_CHAIN:.*]] = arith.addi %[[BASE_BYTES_CHAIN]], %[[BIAS_CHAIN]] : i64
// CHECK: %[[ADDR_ELEMS_CHAIN:.*]] = arith.shrsi %[[ADDR_BYTES_CHAIN]], %{{.*}} : i64
// CHECK: %[[IDX_CHAIN:.*]] = arith.index_cast %[[ADDR_ELEMS_CHAIN]] : i64 to index
// CHECK: dataflow.load %[[MEM_CHAIN]][%[[IDX_CHAIN]]] %arg0 : memref<?xi16>
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
dataflow.graph.func private @g_chained_gep_i8_i16(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr,
    %arg5: i16) -> (none, i16) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %0 = dataflow.carry %rwc, %arg5, %4 : i16
  %1 = llvm.getelementptr %arg4[%index] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %2 = llvm.getelementptr %1[2] : (!llvm.ptr) -> !llvm.ptr, i8
  %3 = llvm.load %2 : !llvm.ptr -> i16
  %4 = arith.addi %0, %3 : i16
  dataflow.graph.return %arg0, %0 : none, i16
}

// An exact negative byte bias must remain negative when converted to an
// element index.

// CHECK-LABEL: dataflow.graph.func private @g_chained_gep_negative_bias
// CHECK-DAG: %[[NEG_MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xi16>
// CHECK: %[[NEG_INDEX:.*]], %[[NEG_RWC:.*]] = dataflow.stream
// CHECK: %[[NEG_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[NEG_BASE_BYTES:.*]] = arith.muli %[[NEG_INDEX]], %[[NEG_STRIDE]] : i64
// CHECK: %[[NEG_BIAS:.*]] = arith.constant -2 : i64
// CHECK: %[[NEG_BYTES:.*]] = arith.addi %[[NEG_BASE_BYTES]], %[[NEG_BIAS]] : i64
// CHECK: %[[NEG_ELEMENTS:.*]] = arith.shrsi %[[NEG_BYTES]], %{{.*}} : i64
// CHECK: %[[NEG_ADDR:.*]] = arith.index_cast %[[NEG_ELEMENTS]] : i64 to index
// CHECK: dataflow.load %[[NEG_MEM]][%[[NEG_ADDR]]] %arg0 : memref<?xi16>
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
dataflow.graph.func private @g_chained_gep_negative_bias(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr)
    -> (none, i16) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %base = llvm.getelementptr %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %ptr = llvm.getelementptr %base[-2]
      : (!llvm.ptr) -> !llvm.ptr, i8
  %value = llvm.load %ptr : !llvm.ptr -> i16
  dataflow.graph.return %arg0, %value : none, i16
}

// Volatile and atomic LLVM accesses retain semantics that dataflow memory ops
// do not represent.

// CHECK-LABEL: dataflow.graph.func private @g_volatile_atomic
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[MEM_INDEX:.*]], %[[MEM_RWC:.*]] = dataflow.stream
// CHECK: %[[MEM_PTR:.*]] = llvm.getelementptr %arg4[%[[MEM_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: %[[VOLATILE_VALUE:.*]] = llvm.load volatile %[[MEM_PTR]] : !llvm.ptr -> i32
// CHECK: llvm.store %arg5, %[[MEM_PTR]] atomic monotonic {alignment = 4 : i64} : i32, !llvm.ptr
// CHECK-NOT: dataflow.load
// CHECK-NOT: dataflow.store
dataflow.graph.func private @g_volatile_atomic(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr,
    %arg5: i32) -> (none, i32) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %ptr = llvm.getelementptr %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %value = llvm.load volatile %ptr : !llvm.ptr -> i32
  llvm.store %arg5, %ptr atomic monotonic {alignment = 4 : i64}
      : i32, !llvm.ptr
  dataflow.graph.return %arg0, %value : none, i32
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

// A preincrement load/store through the carried pointer starts one element
// after the base pointer. The stream ordinal remains zero-based for the
// carried pointer state, so the memory address must use ordinal + 1 while the
// runtime must still provide the memory ctrl tokens for each true item.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_preincrement_i8_f32
// CHECK-DAG: %[[SRC_PRE:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK-DAG: %[[DST_PRE:.*]] = builtin.unrealized_conversion_cast %arg5 : !llvm.ptr to memref<?xf32>
// CHECK: %[[STREAM_PRE:.*]], %[[RWC_PRE:.*]] = dataflow.stream
// CHECK: %[[ZERO_PRE:.*]] = dataflow.constant %arg0 {const_value = 0 : i32} : i32
// CHECK: %[[ONE_PRE:.*]] = dataflow.constant %arg0 {const_value = 1 : i32} : i32
// CHECK: %[[STABLE_ONE_PRE:.*]] = dataflow.invariant %[[RWC_PRE]], %[[ONE_PRE]] : i32
// CHECK: %[[ORD_PRE:.*]] = dataflow.carry %[[RWC_PRE]], %[[ZERO_PRE]], %[[NEXT_PRE:.*]] : i32
// CHECK: %[[NEXT_PRE]] = arith.addi %[[ORD_PRE]], %[[STABLE_ONE_PRE]] : i32
// CHECK: %[[BIAS_PRE:.*]] = dataflow.constant %arg0 {const_value = 1 : i32} : i32
// CHECK: %[[STABLE_BIAS_PRE:.*]] = dataflow.invariant %[[RWC_PRE]], %[[BIAS_PRE]] : i32
// CHECK: %[[ADDR_PRE:.*]] = arith.addi %[[ORD_PRE]], %[[STABLE_BIAS_PRE]] : i32
// CHECK: %[[IDX_PRE:.*]] = arith.index_cast %[[ADDR_PRE]] : i32 to index
// CHECK: dataflow.load %[[SRC_PRE]][%[[IDX_PRE]]] %arg0 : memref<?xf32>
// CHECK: %[[STORE_BIAS_PRE:.*]] = dataflow.constant %arg0 {const_value = 1 : i32} : i32
// CHECK: %[[STORE_STABLE_BIAS_PRE:.*]] = dataflow.invariant %[[RWC_PRE]], %[[STORE_BIAS_PRE]] : i32
// CHECK: %[[STORE_ADDR_PRE:.*]] = arith.addi %[[ORD_PRE]], %[[STORE_STABLE_BIAS_PRE]] : i32
// CHECK: %[[STORE_IDX_PRE:.*]] = arith.index_cast %[[STORE_ADDR_PRE]] : i32 to index
// CHECK: dataflow.store %[[DST_PRE]][%[[STORE_IDX_PRE]]] %{{.*}} %arg0 : memref<?xf32>
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_pointer_carry_preincrement_i8_f32(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: !llvm.ptr,
    %arg5: !llvm.ptr, %bias: f32) -> (none, !llvm.ptr, !llvm.ptr) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i32
  %src_cur = dataflow.carry %rwc, %arg4, %src_next : !llvm.ptr
  %dst_cur = dataflow.carry %rwc, %arg5, %dst_next : !llvm.ptr
  %src_next = llvm.getelementptr %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %data = llvm.load %src_next : !llvm.ptr -> f32
  %sum = arith.addf %data, %bias : f32
  %dst_next = llvm.getelementptr %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %sum, %dst_next : f32, !llvm.ptr
  dataflow.graph.return %arg0, %src_cur, %dst_cur : none, !llvm.ptr, !llvm.ptr
}

// A dynamic GEP from a carried pointer is not a constant per-item bias. Until
// memory lowering can preserve the dynamic index, it must leave the access
// untouched instead of silently lowering it to ordinal + 0.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_dynamic_offset_i8_f32
// CHECK: %[[STREAM_DYN:.*]], %[[RWC_DYN:.*]] = dataflow.stream
// CHECK: %[[SRC_CUR_DYN:.*]] = dataflow.carry %[[RWC_DYN]]
// CHECK: %[[OFFSET_DYN:.*]] = arith.addi %[[STREAM_DYN]], %arg6 : i32
// CHECK: %[[SRC_DYN:.*]] = llvm.getelementptr %[[SRC_CUR_DYN]][%[[OFFSET_DYN]]] : (!llvm.ptr, i32) -> !llvm.ptr, i8
// CHECK: llvm.load %[[SRC_DYN]] : !llvm.ptr -> f32
// CHECK-NOT: dataflow.load
dataflow.graph.func private @g_pointer_carry_dynamic_offset_i8_f32(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: !llvm.ptr,
    %bias: f32, %dyn: i32) -> (none, !llvm.ptr) {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i32
  %src_cur = dataflow.carry %rwc, %arg4, %src_next : !llvm.ptr
  %src_next = llvm.getelementptr %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %offset = arith.addi %index, %dyn : i32
  %src_dyn = llvm.getelementptr %src_cur[%offset] : (!llvm.ptr, i32) -> !llvm.ptr, i8
  %data = llvm.load %src_dyn : !llvm.ptr -> f32
  %sum = arith.addf %data, %bias : f32
  dataflow.graph.return %arg0, %src_cur : none, !llvm.ptr
}

// -----

// A GEP stride that wraps to zero in the LLVM pointer index domain cannot be
// replaced by the unscaled dynamic index.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_index_wrap
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[WRAP_INDEX:.*]], %[[WRAP_RWC:.*]] = dataflow.stream
// CHECK: %[[WRAP_PTR:.*]] = llvm.getelementptr %arg4[%[[WRAP_INDEX]]] : (!llvm.ptr, i8) -> !llvm.ptr, !llvm.array<256 x i8>
// CHECK: llvm.load %[[WRAP_PTR]] : !llvm.ptr -> i8
// CHECK-NOT: dataflow.load
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<!llvm.ptr, dense<[8, 8, 8, 8]> : vector<4xi64>>
>} {
  dataflow.graph.func private @g_pointer_index_wrap(
      %arg0: none, %arg1: i8, %arg2: i8, %arg3: i8, %arg4: !llvm.ptr)
      -> (none, i8) {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        {cont_cond = "<", step_op = "+="} : i8
    %ptr = llvm.getelementptr %arg4[%index]
        : (!llvm.ptr, i8) -> !llvm.ptr, !llvm.array<256 x i8>
    %value = llvm.load %ptr : !llvm.ptr -> i8
    dataflow.graph.return %arg0, %value : none, i8
  }
}

// -----

// A 64-bit address cannot be truncated to a 32-bit MLIR index.

// CHECK-LABEL: dataflow.graph.func private @g_index_truncation
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[WIDE_INDEX:.*]], %[[WIDE_RWC:.*]] = dataflow.stream
// CHECK: %[[WIDE_PTR:.*]] = llvm.getelementptr %arg4[%[[WIDE_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: llvm.load %[[WIDE_PTR]] : !llvm.ptr -> i32
// CHECK-NOT: dataflow.load
module attributes { dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<index, 32>,
  #dlti.dl_entry<!llvm.ptr, dense<[64, 64, 64, 64]> : vector<4xi64>>
>} {
  dataflow.graph.func private @g_index_truncation(
      %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr)
      -> (none, i32) {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        {cont_cond = "<", step_op = "+="} : i64
    %ptr = llvm.getelementptr %arg4[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %ptr : !llvm.ptr -> i32
    dataflow.graph.return %arg0, %value : none, i32
  }
}

// -----

// Graph-memory bridges currently preserve neither non-default address spaces
// nor non-integral pointer semantics.

// CHECK-LABEL: dataflow.graph.func private @g_non_integral_pointer
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[NI_INDEX:.*]], %[[NI_RWC:.*]] = dataflow.stream
// CHECK: %[[NI_PTR:.*]] = llvm.getelementptr %arg4[%[[NI_INDEX]]] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
// CHECK: llvm.load %[[NI_PTR]] : !llvm.ptr<1> -> i32
// CHECK-NOT: dataflow.load
module attributes {
  llvm.data_layout = "e-p:64:64-p1:64:64-ni:1",
  dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<!llvm.ptr<1>, dense<[64, 64, 64, 64]> : vector<4xi64>>
  >
} {
  dataflow.graph.func private @g_non_integral_pointer(
      %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
      %arg4: !llvm.ptr<1>) -> (none, i32) {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        {cont_cond = "<", step_op = "+="} : i64
    %ptr = llvm.getelementptr %arg4[%index]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %value = llvm.load %ptr : !llvm.ptr<1> -> i32
    dataflow.graph.return %arg0, %value : none, i32
  }
}

// -----

// The LLVM data layout is the pointer-index-width authority when present.

// CHECK-LABEL: dataflow.graph.func private @g_llvm_layout_pointer_index
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[LAYOUT_INDEX:.*]], %[[LAYOUT_RWC:.*]] = dataflow.stream
// CHECK: %[[LAYOUT_PTR:.*]] = llvm.getelementptr %arg4[%[[LAYOUT_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// CHECK: llvm.load %[[LAYOUT_PTR]] : !llvm.ptr -> i32
// CHECK-NOT: dataflow.load
module attributes {llvm.data_layout = "e-p:32:32"} {
  dataflow.graph.func private @g_llvm_layout_pointer_index(
      %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr)
      -> (none, i32) {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        {cont_cond = "<", step_op = "+="} : i64
    %ptr = llvm.getelementptr %arg4[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %ptr : !llvm.ptr -> i32
    dataflow.graph.return %arg0, %value : none, i32
  }
}
