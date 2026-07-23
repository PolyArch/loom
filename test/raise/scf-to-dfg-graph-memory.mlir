// These graphs index memory with i64 ordinals, so they need a 64-bit
// canonical index rather than the configured default.
// RUN: env LOOM_INDEX_WIDTH=64 loom-raise-opt --loom-lower-graph-memory \
// RUN:   -split-input-file -verify-diagnostics %s -o %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir

// Explicit graph memory inputs are normalized into canonical dataflow memory
// operations. Their pointer bridge preserves the graph-owned import root.

// CHECK-LABEL: dataflow.graph private @g_canonical
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
dataflow.graph private @g_canonical(%arg0: none, %arg1: i64, %arg2: i64,
                                         %arg3: i64, %arg5: f32,
                                         %arg4: !llvm.ptr) -> (f32)
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

// A no-wrap GEP with element stride retains an element index rather than a
// byte multiply and inverse shift.

// CHECK-LABEL: dataflow.graph private @g_inbounds_element_index
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
dataflow.graph private @g_inbounds_element_index(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: !llvm.ptr) -> ()
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

// Nested accesses retain the graph-scope capability while only the address
// value recurs through the lowered loop.

// CHECK-LABEL: dataflow.graph private @g_nested_static_bridge
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
dataflow.graph private @g_nested_static_bridge(
    %start: none, %lb: i64, %ub: i64, %step: i64, %base: !llvm.ptr) -> ()
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

// Unsigned-only no-wrap remains on the conservative byte-normalization path.

// CHECK-LABEL: dataflow.graph private @g_nuw_element_index
// CHECK: %[[NUW_INDEX:.*]], %[[NUW_RWC:.*]] = dataflow.stream
// CHECK: %[[NUW_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[NUW_BYTES:.*]] = arith.muli %[[NUW_INDEX]], %[[NUW_STRIDE]] : i64
// CHECK: %[[NUW_ELEMENTS:.*]] = arith.shrsi %[[NUW_BYTES]], %{{.*}} : i64
// CHECK: %[[NUW_ADDR:.*]] = arith.index_cast %[[NUW_ELEMENTS]] : i64 to index
dataflow.graph private @g_nuw_element_index(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: !llvm.ptr) -> ()
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

// A zero companion GEP stays on the general byte-normalization path.

// CHECK-LABEL: dataflow.graph private @g_inbounds_zero_companion
// CHECK: %[[CHAIN_ZERO_INDEX:.*]], %[[CHAIN_ZERO_RWC:.*]] = dataflow.stream
// CHECK: %[[CHAIN_ZERO_STRIDE:.*]] = arith.constant 4 : i64
// CHECK: %[[CHAIN_ZERO_BYTES:.*]] = arith.muli %[[CHAIN_ZERO_INDEX]], %[[CHAIN_ZERO_STRIDE]] : i64
// CHECK: %[[CHAIN_ZERO_ELEMENTS:.*]] = arith.shrsi %[[CHAIN_ZERO_BYTES]], %{{.*}} : i64
// CHECK: %[[CHAIN_ZERO_ADDR:.*]] = arith.index_cast %[[CHAIN_ZERO_ELEMENTS]] : i64 to index
dataflow.graph private @g_inbounds_zero_companion(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64,
    %arg4: !llvm.ptr) -> ()
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

// Chained byte offsets preserve exact element conversion, including a
// negative constant bias.

// CHECK-LABEL: dataflow.graph private @g_chained_gep_i8_i16
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
dataflow.graph private @g_chained_gep_i8_i16(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg5: i16,
    %arg4: !llvm.ptr) -> (i16)
    attributes {input_segments = array<i32: 4, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %index, %rwc = dataflow.stream %arg1, %arg2, %arg3
      step add while slt : i64
  %0 = dataflow.carry %rwc, %arg5, %4 : i16
  %1 = llvm.getelementptr %arg4[%index]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
  %2 = llvm.getelementptr %1[2] : (!llvm.ptr) -> !llvm.ptr, i8
  %3 = llvm.load %2 : !llvm.ptr -> i16
  %4 = arith.addi %0, %3 : i16
  dataflow.graph.return %arg0, %0 : none, i16
}

// CHECK-LABEL: dataflow.graph private @g_rank3_row_major(
// CHECK: %[[LD_DIM1:.*]] = dataflow.constant %arg0 {const_value = 5 : index} : index
// CHECK: %[[LD_MUL1:.*]] = arith.muli %arg1, %[[LD_DIM1]] : index
// CHECK: %[[LD_ADD1:.*]] = arith.addi %[[LD_MUL1]], %arg2 : index
// CHECK: %[[LD_DIM2:.*]] = dataflow.constant %arg0 {const_value = 7 : index} : index
// CHECK: %[[LD_MUL2:.*]] = arith.muli %[[LD_ADD1]], %[[LD_DIM2]] : index
// CHECK: %[[LD_ADDR:.*]] = arith.addi %[[LD_MUL2]], %arg3 : index
// CHECK: %[[LD_DATA:.*]], %[[LD_DONE:.*]] = dataflow.load %arg4[%[[LD_ADDR]]] %arg0 : memref<3x5x7xf32>
// CHECK: %[[ST_MUL1:.*]] = arith.muli %arg1, %{{.*}} : index
// CHECK: %[[ST_ADD1:.*]] = arith.addi %[[ST_MUL1]], %arg2 : index
// CHECK: %[[ST_MUL2:.*]] = arith.muli %[[ST_ADD1]], %{{.*}} : index
// CHECK: %[[ST_ADDR:.*]] = arith.addi %[[ST_MUL2]], %arg3 : index
// CHECK: dataflow.store %arg4[%[[ST_ADDR]]] %[[LD_DATA]] %[[LD_DONE]] : memref<3x5x7xf32>
dataflow.graph private @g_rank3_row_major(
    %arg0: none, %arg1: index, %arg2: index, %arg3: index,
    %arg4: memref<3x5x7xf32>) -> ()
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %arg4[%arg1, %arg2, %arg3] : memref<3x5x7xf32>
  memref.store %value, %arg4[%arg1, %arg2, %arg3] : memref<3x5x7xf32>
  dataflow.graph.return %arg0 : none
}

// CHECK-LABEL: dataflow.graph private @g_chained_gep_negative_bias
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
dataflow.graph private @g_chained_gep_negative_bias(
    %arg0: none, %arg1: i64, %arg2: i64, %arg3: i64, %arg4: !llvm.ptr)
    -> (i16)
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

// Rank-zero access uses linear address zero.

// CHECK-LABEL: dataflow.graph private @g_rank0_scalar
// CHECK: %[[R0_LD_ADDR:.*]] = dataflow.constant %arg0 {const_value = 0 : index} : index
// CHECK: %[[R0_DATA:.*]], %[[R0_DONE:.*]] = dataflow.load %arg1[%[[R0_LD_ADDR]]] %arg0 : memref<f32>
// CHECK: %[[R0_ST_ADDR:.*]] = dataflow.constant %arg0 {const_value = 0 : index} : index
// CHECK: dataflow.store %arg1[%[[R0_ST_ADDR]]] %[[R0_DATA]] %[[R0_DONE]] : memref<f32>
dataflow.graph private @g_rank0_scalar(
    %arg0: none, %arg4: memref<f32>) -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %arg4[] : memref<f32>
  memref.store %value, %arg4[] : memref<f32>
  dataflow.graph.return %arg0 : none
}

// -----

// Dynamic rank-one access keeps its sole index unchanged.

// CHECK-LABEL: dataflow.graph private @g_dynamic_rank1
// CHECK: %[[D1_DATA:.*]], %[[D1_DONE:.*]] = dataflow.load %arg2[%arg1] %arg0 : memref<?xf32>
// CHECK: dataflow.store %arg2[%arg1] %[[D1_DATA]] %[[D1_DONE]] : memref<?xf32>
dataflow.graph private @g_dynamic_rank1(
    %arg0: none, %arg1: index, %arg4: memref<?xf32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %arg4[%arg1] : memref<?xf32>
  memref.store %value, %arg4[%arg1] : memref<?xf32>
  dataflow.graph.return %arg0 : none
}

// -----

// Rank greater than one requires a static identity layout.
dataflow.graph private @g_static_nonidentity(
    %arg0: none, %arg1: index, %arg2: index,
    %arg4: memref<4x8xf32, affine_map<(d0, d1) -> (d1, d0)>>) -> ()
    attributes {input_segments = array<i32: 2, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  // expected-error @+1 {{loom-lower-graph-memory: memref.load requires an identity-layout memref whose shape is static when rank exceeds one}}
  %value = memref.load %arg4[%arg1, %arg2] : memref<4x8xf32, affine_map<(d0, d1) -> (d1, d0)>>
  memref.store %value, %arg4[%arg1, %arg2] : memref<4x8xf32, affine_map<(d0, d1) -> (d1, d0)>>
  dataflow.graph.return %arg0 : none
}

// -----

dataflow.graph private @g_dynamic_rank2(
    %arg0: none, %arg1: index, %arg2: index, %arg4: memref<?x?xf32>) -> ()
    attributes {input_segments = array<i32: 2, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  // expected-error @+1 {{loom-lower-graph-memory: memref.load requires an identity-layout memref whose shape is static when rank exceeds one}}
  %value = memref.load %arg4[%arg1, %arg2] : memref<?x?xf32>
  memref.store %value, %arg4[%arg1, %arg2] : memref<?x?xf32>
  dataflow.graph.return %arg0 : none
}

// -----

// The maximum address is computed without truncating the 65-bit product.
dataflow.graph private @g_address_beyond_64_bits(
    %arg0: none, %arg1: index, %arg2: index,
    %arg4: memref<8589934592x4294967296xi8>) -> ()
    attributes {input_segments = array<i32: 2, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  // expected-error @+1 {{loom-lower-graph-memory: maximum linear address 36893488147419103231 is not representable in the graph's resolved signed index domain 'i64'}}
  %value = memref.load %arg4[%arg1, %arg2] : memref<8589934592x4294967296xi8>
  memref.store %value, %arg4[%arg1, %arg2] : memref<8589934592x4294967296xi8>
  dataflow.graph.return %arg0 : none
}
