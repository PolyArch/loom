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

// A direct no-wrap GEP whose stride equals the accessed element width already
// carries an element index. Preserve that index instead of materializing a
// multiply-by-width followed by the inverse element-width shift.

// CHECK-LABEL: dataflow.graph.func private @g_inbounds_element_index
// CHECK-DAG: %[[INBOUNDS_MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xi32>
// CHECK: %[[INBOUNDS_INDEX:.*]], %[[INBOUNDS_RWC:.*]] = dataflow.stream
// CHECK-NOT: arith.muli
// CHECK-NOT: arith.shrsi
// CHECK: %[[INBOUNDS_ADDR:.*]] = arith.index_cast %[[INBOUNDS_INDEX]] : i64 to index
// CHECK: %[[INBOUNDS_DATA:.*]], %[[INBOUNDS_DONE:.*]] = dataflow.load %[[INBOUNDS_MEM]][%[[INBOUNDS_ADDR]]] %arg0 : memref<?xi32>
// CHECK: dataflow.store %[[INBOUNDS_MEM]][%[[INBOUNDS_ADDR]]] %[[INBOUNDS_DATA]] %arg0 : memref<?xi32>
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_inbounds_element_index(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: !llvm.ptr) -> none {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %ptr = llvm.getelementptr inbounds %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %value = llvm.load %ptr : !llvm.ptr -> i32
  llvm.store %value, %ptr : i32, !llvm.ptr
  dataflow.graph.return %arg0 : none
}

// Unsigned-only no-wrap does not prove that signed byte multiplication can be
// inverted. Keep the conservative byte-domain normalization.

// CHECK-LABEL: dataflow.graph.func private @g_nuw_element_index
// CHECK: %[[NUW_INDEX:.*]], %[[NUW_RWC:.*]] = dataflow.stream
// CHECK: %[[NUW_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[NUW_BYTES:.*]] = arith.muli %[[NUW_INDEX]], %[[NUW_STRIDE]] : i64
// CHECK: %[[NUW_ELEMENTS:.*]] = arith.shrsi %[[NUW_BYTES]], %{{.*}} : i64
// CHECK: %[[NUW_ADDR:.*]] = arith.index_cast %[[NUW_ELEMENTS]] : i64 to index
dataflow.graph.func private @g_nuw_element_index(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: !llvm.ptr) -> none {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %ptr = llvm.getelementptr nuw %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %value = llvm.load %ptr : !llvm.ptr -> i32
  llvm.store %value, %ptr : i32, !llvm.ptr
  dataflow.graph.return %arg0 : none
}

// Keep chained GEPs on the general byte-normalization path even when the
// companion offset is zero.

// CHECK-LABEL: dataflow.graph.func private @g_inbounds_zero_companion
// CHECK: %[[CHAIN_ZERO_INDEX:.*]], %[[CHAIN_ZERO_RWC:.*]] = dataflow.stream
// CHECK: %[[CHAIN_ZERO_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[CHAIN_ZERO_BYTES:.*]] = arith.muli %[[CHAIN_ZERO_INDEX]], %[[CHAIN_ZERO_STRIDE]] : i64
// CHECK: %[[CHAIN_ZERO_ELEMENTS:.*]] = arith.shrsi %[[CHAIN_ZERO_BYTES]], %{{.*}} : i64
// CHECK: %[[CHAIN_ZERO_ADDR:.*]] = arith.index_cast %[[CHAIN_ZERO_ELEMENTS]] : i64 to index
dataflow.graph.func private @g_inbounds_zero_companion(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: !llvm.ptr) -> none {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      {cont_cond = "<", step_op = "+="} : i64
  %base = llvm.getelementptr inbounds %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %ptr = llvm.getelementptr %base[0]
      : (!llvm.ptr) -> !llvm.ptr, i8
  %value = llvm.load %ptr : !llvm.ptr -> i32
  llvm.store %value, %ptr : i32, !llvm.ptr
  dataflow.graph.return %arg0 : none
}

// Negative-bail #1: a graph.func body whose llvm.load / llvm.store
// use a base pointer derived from a global address-of (not a graph
// block-arg) keeps the original llvm.{load, store, gep} chain.

llvm.mlir.global private @global_buf(dense<0.0> : tensor<8xf32>) : !llvm.array<8 x f32>

// CHECK-LABEL: dataflow.graph.func private @g_global_base
// CHECK: %[[GLOBAL_IV:.*]], %{{.*}} = dataflow.stream
// CHECK: %[[GBL:.*]] = llvm.mlir.addressof @global_buf
// CHECK: %[[GEP:.*]] = llvm.getelementptr %[[GBL]][%[[GLOBAL_IV]]]
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
// CHECK: %[[BYTE_IV:.*]], %{{.*}} = dataflow.stream
// CHECK: %[[BYTE:.*]] = arith.shli %[[BYTE_IV]], %arg5 : i64
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

// -----

module attributes {
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {

// Pointer induction through a carried LLVM pointer is a memory-view concern,
// not a fabric pointer operation. The stream IV already has exactly K tokens,
// so memory lowering uses it directly and does not synthesize a second carry
// recurrence or gate the IV with its phase.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_i8_f32
// CHECK-DAG: %[[SRC:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK-DAG: %[[DST:.*]] = builtin.unrealized_conversion_cast %arg5 : !llvm.ptr to memref<?xf32>
// CHECK: %[[IV:.*]], %[[PHASE:.*]] = dataflow.stream
// CHECK: %[[SRC_RAW:.*]] = dataflow.carry %[[PHASE]], %arg4,
// CHECK: %{{.*}}, %[[SRC_BODY:.*]] = dataflow.gate %[[PHASE]], %[[SRC_RAW]] : !llvm.ptr
// CHECK: %[[DST_RAW:.*]] = dataflow.carry %[[PHASE]], %arg5,
// CHECK: %{{.*}}, %[[DST_BODY:.*]] = dataflow.gate %[[PHASE]], %[[DST_RAW]] : !llvm.ptr
// CHECK-NOT: dataflow.gate %[[PHASE]], %[[IV]] : i32
// CHECK: %[[IDX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK: dataflow.load %[[SRC]][%[[IDX]]] %arg0 : memref<?xf32>
// CHECK: dataflow.store %[[DST]][%[[IDX]]] %{{.*}} %arg0 : memref<?xf32>
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_pointer_carry_i8_f32(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: !llvm.ptr,
    %arg5: !llvm.ptr, %bias: f32) -> (none, !llvm.ptr, !llvm.ptr) {
  %stream_init = arith.constant 0 : i32
  %stream_step = arith.constant 1 : i32
  %index, %rwc = dataflow.stream %stream_init, %arg2, %stream_step
      {cont_cond = "<", step_op = "+="} : i32
  %src_raw = dataflow.carry %rwc, %arg4, %src_next : !llvm.ptr
  %src_phase, %src_cur = dataflow.gate %rwc, %src_raw : !llvm.ptr
  %src_exit:2 = dataflow.demux %rwc, %src_raw
      : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %dst_raw = dataflow.carry %rwc, %arg5, %dst_next : !llvm.ptr
  %dst_phase, %dst_cur = dataflow.gate %rwc, %dst_raw : !llvm.ptr
  %dst_exit:2 = dataflow.demux %rwc, %dst_raw
      : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %src_next = llvm.getelementptr %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %data = llvm.load %src_cur : !llvm.ptr -> f32
  %sum = arith.addf %data, %bias : f32
  %dst_next = llvm.getelementptr %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %sum, %dst_cur : f32, !llvm.ptr
  dataflow.graph.return %arg0, %src_exit#0, %dst_exit#0
      : none, !llvm.ptr, !llvm.ptr
}

// A preincrement load/store uses stream IV plus a projected invariant bias.
// Generated invariant outputs remain in the parent domain until a gate
// projects them into the K-cardinality body domain.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_preincrement_i8_f32
// CHECK-DAG: %[[SRC_PRE:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xf32>
// CHECK-DAG: %[[DST_PRE:.*]] = builtin.unrealized_conversion_cast %arg5 : !llvm.ptr to memref<?xf32>
// CHECK: %[[IV_PRE:.*]], %[[PHASE_PRE:.*]] = dataflow.stream
// CHECK-NOT: dataflow.gate %[[PHASE_PRE]], %[[IV_PRE]] : i32
// CHECK: %[[BIAS_PRE:.*]] = dataflow.constant %arg0 {const_value = 1 : i32} : i32
// CHECK: %[[STABLE_BIAS_RAW_PRE:.*]] = dataflow.invariant %[[PHASE_PRE]], %[[BIAS_PRE]] : i32
// CHECK: %{{.*}}, %[[STABLE_BIAS_PRE:.*]] = dataflow.gate %[[PHASE_PRE]], %[[STABLE_BIAS_RAW_PRE]] : i32
// CHECK: %[[ADDR_PRE:.*]] = arith.addi %[[IV_PRE]], %[[STABLE_BIAS_PRE]] : i32
// CHECK: %[[IDX_PRE:.*]] = arith.index_cast %[[ADDR_PRE]] : i32 to index
// CHECK: dataflow.load %[[SRC_PRE]][%[[IDX_PRE]]] %arg0 : memref<?xf32>
// CHECK: %[[STORE_BIAS_PRE:.*]] = dataflow.constant %arg0 {const_value = 1 : i32} : i32
// CHECK: %[[STORE_STABLE_BIAS_RAW_PRE:.*]] = dataflow.invariant %[[PHASE_PRE]], %[[STORE_BIAS_PRE]] : i32
// CHECK: %{{.*}}, %[[STORE_STABLE_BIAS_PRE:.*]] = dataflow.gate %[[PHASE_PRE]], %[[STORE_STABLE_BIAS_RAW_PRE]] : i32
// CHECK: %[[STORE_ADDR_PRE:.*]] = arith.addi %[[IV_PRE]], %[[STORE_STABLE_BIAS_PRE]] : i32
// CHECK: %[[STORE_IDX_PRE:.*]] = arith.index_cast %[[STORE_ADDR_PRE]] : i32 to index
// CHECK: dataflow.store %[[DST_PRE]][%[[STORE_IDX_PRE]]] %{{.*}} %arg0 : memref<?xf32>
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_pointer_carry_preincrement_i8_f32(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: !llvm.ptr,
    %arg5: !llvm.ptr, %bias: f32) -> (none, !llvm.ptr, !llvm.ptr) {
  %stream_init = arith.constant 0 : i32
  %stream_step = arith.constant 1 : i32
  %index, %rwc = dataflow.stream %stream_init, %arg2, %stream_step
      {cont_cond = "<", step_op = "+="} : i32
  %src_raw = dataflow.carry %rwc, %arg4, %src_next : !llvm.ptr
  %src_phase, %src_cur = dataflow.gate %rwc, %src_raw : !llvm.ptr
  %src_exit:2 = dataflow.demux %rwc, %src_raw
      : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %dst_raw = dataflow.carry %rwc, %arg5, %dst_next : !llvm.ptr
  %dst_phase, %dst_cur = dataflow.gate %rwc, %dst_raw : !llvm.ptr
  %dst_exit:2 = dataflow.demux %rwc, %dst_raw
      : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %src_next = llvm.getelementptr %src_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %data = llvm.load %src_next : !llvm.ptr -> f32
  %sum = arith.addf %data, %bias : f32
  %dst_next = llvm.getelementptr %dst_cur[4] : (!llvm.ptr) -> !llvm.ptr, i8
  llvm.store %sum, %dst_next : f32, !llvm.ptr
  dataflow.graph.return %arg0, %src_exit#0, %dst_exit#0
      : none, !llvm.ptr, !llvm.ptr
}

// A unit-stride pointer carry does not make an arbitrary recurrence IV an
// iteration ordinal. With a nonzero stream init, memory lowering must retain
// the carried pointer address instead of replacing it with base + stream.iv.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_nonordinal_init
// CHECK: %[[IV_NONORD:.*]], %[[PHASE_NONORD:.*]] = dataflow.stream
// CHECK: %[[RAW_NONORD:.*]] = dataflow.carry %[[PHASE_NONORD]], %arg3,
// CHECK: %{{.*}}, %[[CUR_NONORD:.*]] = dataflow.gate %[[PHASE_NONORD]], %[[RAW_NONORD]] : !llvm.ptr
// CHECK-NOT: arith.index_cast %[[IV_NONORD]]
// CHECK: llvm.load %[[CUR_NONORD]] : !llvm.ptr -> f32
// CHECK-NOT: dataflow.load
dataflow.graph.func private @g_pointer_carry_nonordinal_init(
    %arg0: none, %arg1: i32, %arg2: i32, %arg3: !llvm.ptr,
    %bias: f32) -> (none, !llvm.ptr) {
  %stream_init = arith.constant 4 : i32
  %stream_step = arith.constant 1 : i32
  %index, %phase = dataflow.stream %stream_init, %arg1, %stream_step
      {cont_cond = "<", step_op = "+="} : i32
  %raw = dataflow.carry %phase, %arg3, %next : !llvm.ptr
  %body_phase, %current = dataflow.gate %phase, %raw : !llvm.ptr
  %exit:2 = dataflow.demux %phase, %raw
      : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %data = llvm.load %current : !llvm.ptr -> f32
  %sum = arith.addf %data, %bias : f32
  %next = llvm.getelementptr %current[4]
      : (!llvm.ptr) -> !llvm.ptr, i8
  dataflow.graph.return %arg0, %exit#0 : none, !llvm.ptr
}

}

// -----

// Widening an i8 recurrence to the target index type does not preserve the
// pointer-carry ordinal after the recurrence wraps. Keep the carried pointer
// address instead of rewriting it as base + stream.iv.

// CHECK-LABEL: dataflow.graph.func private @g_pointer_carry_widening_i8
// CHECK: %[[IV_I8:.*]], %[[PHASE_I8:.*]] = dataflow.stream
// CHECK: %[[RAW_I8:.*]] = dataflow.carry %[[PHASE_I8]], %arg2,
// CHECK: %{{.*}}, %[[CUR_I8:.*]] = dataflow.gate %[[PHASE_I8]], %[[RAW_I8]] : !llvm.ptr
// CHECK-NOT: arith.index_cast %[[IV_I8]] : i8 to index
// CHECK: llvm.load %[[CUR_I8]] : !llvm.ptr -> i8
// CHECK-NOT: dataflow.load
dataflow.graph.func private @g_pointer_carry_widening_i8(
    %arg0: none, %arg1: i8, %arg2: !llvm.ptr) -> (none, !llvm.ptr) {
  %stream_init = arith.constant 0 : i8
  %stream_step = arith.constant 1 : i8
  %iv, %phase = dataflow.stream %stream_init, %arg1, %stream_step
      {cont_cond = "!=", step_op = "+="} : i8
  %raw = dataflow.carry %phase, %arg2, %next : !llvm.ptr
  %body_phase, %current = dataflow.gate %phase, %raw : !llvm.ptr
  %exit:2 = dataflow.demux %phase, %raw
      : (i1, !llvm.ptr) -> (!llvm.ptr, !llvm.ptr)
  %value = llvm.load %current : !llvm.ptr -> i8
  %next = llvm.getelementptr %current[1]
      : (!llvm.ptr) -> !llvm.ptr, i8
  dataflow.graph.return %arg0, %exit#0 : none, !llvm.ptr
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
