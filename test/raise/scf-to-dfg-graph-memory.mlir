// RUN: loom-raise-opt --split-input-file --loom-lower-graph-memory %s | FileCheck %s

// Positive case: a graph.func body with the canonical
// (dataflow.stream + dataflow.carry + llvm.gep + llvm.load + arith +
// llvm.store) shape gets the residual memory ops tokenized into
// dataflow.load / dataflow.store. The graph block-arg !llvm.ptr is
// bridged to memref<?xf32> via builtin.unrealized_conversion_cast. Address
// arithmetic remains in the LLVM pointer-index width until the exact byte
// offset is converted to an element index.

// CHECK-LABEL: dataflow.graph.func private @g_canonical
// CHECK-DAG: %[[MEM:.*]] = builtin.unrealized_conversion_cast %arg5 : !llvm.ptr to memref<?xf32>
// CHECK: %[[STREAM:.*]], %[[RWC:.*]] = dataflow.stream
// CHECK: %[[LOAD_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[LOAD_BYTE:.*]] = arith.muli %[[STREAM]], %[[LOAD_STRIDE]] : i64
// CHECK: %[[LOAD_ELEM:.*]] = arith.shrsi %[[LOAD_BYTE]], %{{.*}} : i64
// CHECK: %[[LOAD_IDX:.*]] = arith.index_cast %[[LOAD_ELEM]] : i64 to index
// CHECK: %{{.*}}, %[[LOAD_DONE:.*]] = dataflow.load %[[MEM]][%[[LOAD_IDX]]] %arg0 : memref<?xf32>
// CHECK: dataflow.store %[[MEM]][%[[LOAD_IDX]]] %{{.*}} %[[LOAD_DONE]] : memref<?xf32>
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_canonical(%arg0: none, %arg1: i64, %arg2: i64,
                                         %arg3: i64, %arg5: f32,
                                         %arg4: !llvm.ptr) -> (none, f32)
    attributes {input_segments = array<i32: 4, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
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
// CHECK: dataflow.store %[[INBOUNDS_MEM]][%[[INBOUNDS_ADDR]]] %[[INBOUNDS_DATA]] %[[INBOUNDS_DONE]] : memref<?xi32>
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_inbounds_element_index(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: !llvm.ptr) -> none
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %ptr = llvm.getelementptr inbounds %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %value = llvm.load %ptr : !llvm.ptr -> i32
  llvm.store %value, %ptr : i32, !llvm.ptr
  dataflow.graph.return %arg0 : none
}

// Nested graph-root accesses retain the memory capability at graph scope.
// Only the address value recurs through the lowered loop.

// CHECK-LABEL: dataflow.graph.func private @g_nested_static_bridge
// CHECK-DAG: %[[NESTED_MEM:.*]] = builtin.unrealized_conversion_cast %arg4 : !llvm.ptr to memref<?xi32>
// CHECK: %[[NESTED_IV:.*]], %[[NESTED_PHASE:.*]] = dataflow.stream
// CHECK-NOT: dataflow.invariant {{.*}} : !llvm.ptr
// CHECK-NOT: dataflow.demux {{.*}} : (i1, !llvm.ptr)
// CHECK: %[[NESTED_ADDR:.*]] = arith.index_cast %[[NESTED_IV]] : i64 to index
// CHECK: %[[NESTED_VALUE:.*]], %[[NESTED_DONE:.*]] = dataflow.load %[[NESTED_MEM]][%[[NESTED_ADDR]]]
// CHECK: dataflow.store %[[NESTED_MEM]][%[[NESTED_ADDR]]] %[[NESTED_VALUE]] %{{.*}}
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
// CHECK-NOT: llvm.store
dataflow.graph.func private @g_nested_static_bridge(
    %start: none, %lb: i64, %ub: i64, %step: i64, %base: !llvm.ptr) -> none
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  scf.for %i = %lb to %ub step %step : i64 {
    %ptr = llvm.getelementptr inbounds %base[%i]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %value = llvm.load %ptr : !llvm.ptr -> i32
    llvm.store %value, %ptr : i32, !llvm.ptr
  }
  dataflow.graph.return %start : none
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
    %arg4: !llvm.ptr) -> none
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
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
    %arg4: !llvm.ptr) -> none
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %base = llvm.getelementptr inbounds %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %ptr = llvm.getelementptr %base[0]
      : (!llvm.ptr) -> !llvm.ptr, i8
  %value = llvm.load %ptr : !llvm.ptr -> i32
  llvm.store %value, %ptr : i32, !llvm.ptr
  dataflow.graph.return %arg0 : none
}

// A dynamic aggregate-stride GEP followed by a constant byte GEP can be
// normalized when both byte contributions convert exactly to load elements.

// CHECK-LABEL: dataflow.graph.func private @g_chained_gep_i8_i16
// CHECK-DAG: %[[MEM_CHAIN:.*]] = builtin.unrealized_conversion_cast %arg5 : !llvm.ptr to memref<?xi16>
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
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg5: i16,
    %arg4: !llvm.ptr) -> (none, i16)
    attributes {input_segments = array<i32: 4, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
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
    -> (none, i16)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %base = llvm.getelementptr %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %ptr = llvm.getelementptr %base[-2]
      : (!llvm.ptr) -> !llvm.ptr, i8
  %value = llvm.load %ptr : !llvm.ptr -> i16
  dataflow.graph.return %arg0, %value : none, i16
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
      -> (none, i8)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        step add while slt : i8
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
      -> (none, i32)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        step add while slt : i64
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
      %arg4: !llvm.ptr<1>) -> (none, i32)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        step add while slt : i64
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
      -> (none, i32)
      attributes {input_segments = array<i32: 3, 0, 1>,
                  result_segments = array<i32: 1, 0, 0>} {
    %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
        step add while slt : i64
    %ptr = llvm.getelementptr %arg4[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, i32
    %value = llvm.load %ptr : !llvm.ptr -> i32
    dataflow.graph.return %arg0, %value : none, i32
  }
}
