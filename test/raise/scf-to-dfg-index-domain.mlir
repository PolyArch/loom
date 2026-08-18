// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// An observable recurrence remains in its source semantic width even when its
// body projection feeds an address.
// CHECK-LABEL: dataflow.graph private @preserve_observable_wide_carry
// CHECK: %[[INIT:.*]] = dataflow.constant %arg0 {const_value = 4294967297 : i64} : i64
// CHECK-NOT: const_value = 4294967297 : index
// CHECK: %[[CARRY:.*]] = dataflow.carry %{{.*}}, %[[INIT]], %[[NEXT:.*]] : i64
// CHECK: %[[EXIT:.*]]:2 = dataflow.demux %{{.*}}, %[[CARRY]] : (i1, i64) -> (i64, i64)
// CHECK: %[[ADDR:.*]] = arith.index_cast %{{.*}} : i64 to index
// CHECK: dataflow.load %arg5[%[[ADDR]]]
// CHECK: %[[NEXT]] = arith.addi %{{.*}}, %{{.*}} : i64
// CHECK: %[[PUBLISHED:.*]]:2 = dataflow.sync %{{.*}}, %[[EXIT]]#0 : (none, i64) -> (none, i64)
// CHECK: dataflow.graph.return values(%[[PUBLISHED]]#1, %{{.*}} : i64, f32)
dataflow.graph private @preserve_observable_wide_carry(
    %start: none, %init: i64, %limit: i64, %step: i64,
    %seed: f32, %memory: memref<?xf32>) -> (i64, f32)
    attributes {input_segments = array<i32: 4, 0, 1>,
                result_segments = array<i32: 2, 0, 0>} {
  %index, %phase = dataflow.stream %init, %limit, %step
      step add while slt : i64
  %step_raw = dataflow.invariant %phase, %step : i64
  %step_phase, %step_body = dataflow.gate %phase, %step_raw : i64
  %cursor0 = dataflow.constant %start {const_value = 4294967297 : i64} : i64
  %cursor_raw = dataflow.carry %phase, %cursor0, %next : i64
  %cursor_phase, %cursor = dataflow.gate %phase, %cursor_raw : i64
  %cursor_exit:2 = dataflow.demux %phase, %cursor_raw
      : (i1, i64) -> (i64, i64)
  %addr = arith.index_cast %cursor : i64 to index
  %data, %done = dataflow.load %memory[%addr] %start : memref<?xf32>
  %next = arith.addi %cursor, %step_body : i64
  dataflow.graph.return %done, %cursor_exit#0, %data : none, i64, f32
}

// Failed carry materialization must erase the speculative index constant and
// preserve the original i64 recurrence.
// CHECK-LABEL: dataflow.graph private @rollback_partial_carry
// CHECK: dataflow.constant %arg0 {const_value = 4 : i64} : i64
// CHECK-NOT: dataflow.constant %arg0 {const_value = 4 : index} : index
// CHECK: dataflow.carry %{{.*}} : i64
// CHECK-NOT: dataflow.carry %{{.*}} : index
dataflow.graph private @rollback_partial_carry(
    %start: none, %init: i64, %limit: i64, %step: i64,
    %memory: memref<?xf32>) -> (i64, f32)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 2, 0, 0>} {
  %index, %phase = dataflow.stream %init, %limit, %step
      step add while slt : i64
  %cursor0 = dataflow.constant %start {const_value = 4 : i64} : i64
  %step_raw = dataflow.invariant %phase, %step : i64
  %unprojected = arith.addi %step_raw, %step : i64
  %step_phase, %step_body = dataflow.gate %phase, %step_raw : i64
  %cursor_raw = dataflow.carry %phase, %cursor0, %next : i64
  %cursor_phase, %cursor = dataflow.gate %phase, %cursor_raw : i64
  %cursor_exit:2 = dataflow.demux %phase, %cursor_raw
      : (i1, i64) -> (i64, i64)
  %addr = arith.index_cast %cursor : i64 to index
  %data, %done = dataflow.load %memory[%addr] %start : memref<?xf32>
  %next = arith.addi %cursor, %step_body : i64
  dataflow.graph.return %done, %cursor_exit#0, %data : none, i64, f32
}

// Recursive address materialization must roll back all converted producers
// when one operand is an unsupported carry.
// CHECK-LABEL: dataflow.graph private @rollback_partial_address
// CHECK: dataflow.constant %arg0 {const_value = 4 : i64} : i64
// CHECK-NOT: dataflow.constant %arg0 {const_value = 4 : index} : index
// CHECK-NOT: arith.addi {{.*}} : index
// CHECK: arith.index_cast %{{.*}} : i64 to index
dataflow.graph private @rollback_partial_address(
    %start: none, %phase: i1, %init: i64, %next: i64,
    %memory: memref<?xf32>) -> (f32)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %base = dataflow.constant %start {const_value = 4 : i64} : i64
  %carried = dataflow.carry %phase, %init, %next : i64
  %sum = arith.addi %base, %carried : i64
  %body_phase, %body_value = dataflow.gate %phase, %sum : i64
  %addr = arith.index_cast %body_value : i64 to index
  %data, %done = dataflow.load %memory[%addr] %start : memref<?xf32>
  dataflow.graph.return %done, %data : none, f32
}

// A comparison must roll back a materialized lhs when the rhs cannot enter
// the index domain.
// CHECK-LABEL: dataflow.graph private @rollback_partial_compare
// CHECK: %[[CMP_LHS:.*]] = dataflow.constant %arg0 {const_value = 4 : i64} : i64
// CHECK: %[[CMP_RHS:.*]] = dataflow.carry %arg1, %arg2, %arg3 : i64
// CHECK-NOT: dataflow.constant %arg0 {const_value = 4 : index} : index
// CHECK: arith.cmpi slt, %[[CMP_LHS]], %[[CMP_RHS]] : i64
// CHECK-NOT: arith.cmpi {{.*}} : index
dataflow.graph private @rollback_partial_compare(
    %start: none, %phase: i1, %init: i64, %next: i64,
    %memory: memref<?xf32>) -> (f32)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %lhs = dataflow.constant %start {const_value = 4 : i64} : i64
  %rhs = dataflow.carry %phase, %init, %next : i64
  %addr = arith.index_cast %rhs : i64 to index
  %data, %done = dataflow.load %memory[%addr] %start : memref<?xf32>
  %predicate = arith.cmpi slt, %lhs, %rhs : i64
  %body_phase, %body_data = dataflow.gate %predicate, %data : f32
  dataflow.graph.return %done, %body_data : none, f32
}

// Address-only arithmetic is narrowed without cloning stateful actors. Their
// source-width results remain the unique owners of reset and retirement.
// CHECK-LABEL: dataflow.graph private @narrow_address_mask
// CHECK: %[[MASK_INDEX:.*]], %[[MASK_PHASE:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[MASK_OFFSET_RAW:.*]] = dataflow.invariant %[[MASK_PHASE]], %arg4 : i64
// CHECK: %{{.*}}, %[[MASK_OFFSET_WIDE:.*]] = dataflow.gate %[[MASK_PHASE]], %[[MASK_OFFSET_RAW]] : i64
// CHECK: %[[MASK_RAW:.*]] = dataflow.invariant %[[MASK_PHASE]], %arg5 : i64
// CHECK: %{{.*}}, %[[MASK_WIDE:.*]] = dataflow.gate %[[MASK_PHASE]], %[[MASK_RAW]] : i64
// CHECK: %[[MASK_IV:.*]] = arith.index_cast %[[MASK_INDEX]] : i64 to index
// CHECK: %[[MASK_OFFSET:.*]] = arith.index_cast %[[MASK_OFFSET_WIDE]] : i64 to index
// CHECK: %[[MASK_ADD:.*]] = arith.addi %[[MASK_IV]], %[[MASK_OFFSET]] : index
// CHECK: %[[MASK_VALUE:.*]] = arith.index_cast %[[MASK_WIDE]] : i64 to index
// CHECK: %[[MASK_ADDR:.*]] = arith.andi %[[MASK_ADD]], %[[MASK_VALUE]] : index
// CHECK: dataflow.load %arg6[%[[MASK_ADDR]]]
dataflow.graph private @narrow_address_mask(
    %start: none, %lower: i64, %upper: i64, %step: i64,
    %offset: i64, %mask: i64, %memory: memref<?xf32>) -> (f32)
    attributes {input_segments = array<i32: 5, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %index, %phase = dataflow.stream %lower, %upper, %step
      step add while slt : i64
  %offset_raw = dataflow.invariant %phase, %offset : i64
  %offset_phase, %offset_body = dataflow.gate %phase, %offset_raw : i64
  %mask_raw = dataflow.invariant %phase, %mask : i64
  %mask_phase, %mask_body = dataflow.gate %phase, %mask_raw : i64
  %biased = arith.addi %index, %offset_body : i64
  %wrapped = arith.andi %biased, %mask_body : i64
  %addr = arith.index_cast %wrapped : i64 to index
  %data, %done = dataflow.load %memory[%addr] %start : memref<?xf32>
  dataflow.graph.return %done, %data : none, f32
}

// Guarded address arithmetic, comparison, and select share one index domain.
// CHECK-LABEL: dataflow.graph private @narrow_guarded_address
// CHECK: %[[GUARD_INDEX:.*]], %[[GUARD_PHASE:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[GUARD_LB_RAW:.*]] = dataflow.invariant %[[GUARD_PHASE]], %arg4 : i64
// CHECK: %{{.*}}, %[[GUARD_LB_WIDE:.*]] = dataflow.gate %[[GUARD_PHASE]], %[[GUARD_LB_RAW]] : i64
// CHECK: %[[GUARD_UB_RAW:.*]] = dataflow.invariant %[[GUARD_PHASE]], %arg5 : i64
// CHECK: %{{.*}}, %[[GUARD_UB_WIDE:.*]] = dataflow.gate %[[GUARD_PHASE]], %[[GUARD_UB_RAW]] : i64
// CHECK: %[[GUARD_UB:.*]] = arith.index_cast %[[GUARD_UB_WIDE]] : i64 to index
// CHECK: %[[GUARD_IV:.*]] = arith.index_cast %[[GUARD_INDEX]] : i64 to index
// CHECK: %[[GUARD_DELTA:.*]] = arith.subi %[[GUARD_UB]], %[[GUARD_IV]] : index
// CHECK: %[[GUARD_LB:.*]] = arith.index_cast %[[GUARD_LB_WIDE]] : i64 to index
// CHECK: %[[GUARD_PRED:.*]] = arith.cmpi sgt, %[[GUARD_DELTA]], %[[GUARD_LB]] : index
// CHECK: %[[GUARD_SAFE:.*]] = arith.select %[[GUARD_PRED]], %[[GUARD_DELTA]], %{{.*}} : index
// CHECK: dataflow.load %arg6[%[[GUARD_SAFE]]]
dataflow.graph private @narrow_guarded_address(
    %start: none, %lower: i64, %upper: i64, %step: i64,
    %lower_guard: i64, %upper_guard: i64, %memory: memref<?xf32>)
    -> (f32)
    attributes {input_segments = array<i32: 5, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %index, %phase = dataflow.stream %lower, %upper, %step
      step add while slt : i64
  %lower_raw = dataflow.invariant %phase, %lower_guard : i64
  %lower_phase, %lower_body = dataflow.gate %phase, %lower_raw : i64
  %upper_raw = dataflow.invariant %phase, %upper_guard : i64
  %upper_phase, %upper_body = dataflow.gate %phase, %upper_raw : i64
  %delta = arith.subi %upper_body, %index : i64
  %predicate = arith.cmpi sgt, %delta, %lower_body : i64
  %addr = arith.index_cast %delta : i64 to index
  %zero = dataflow.constant %start {const_value = 0 : index} : index
  %safe = arith.select %predicate, %addr, %zero : index
  %data, %done = dataflow.load %memory[%safe] %start : memref<?xf32>
  dataflow.graph.return %done, %data : none, f32
}

// A nonnegative i32 expression widened with arith.extui remains address-domain.
// CHECK-LABEL: dataflow.graph private @narrow_zext_address
// CHECK: %[[ZEXT_LHS:.*]] = arith.index_cast %arg4 : i32 to index
// CHECK: %[[ZEXT_RHS:.*]] = arith.index_cast %arg5 : i32 to index
// CHECK: %[[ZEXT_ADDR:.*]] = arith.addi %[[ZEXT_LHS]], %[[ZEXT_RHS]] : index
// CHECK-NOT: arith.extui
// CHECK: dataflow.load %arg6[%[[ZEXT_ADDR]]]
dataflow.graph private @narrow_zext_address(
    %start: none, %lower: i64, %upper: i64, %step: i64,
    %lhs: i32, %rhs: i32, %memory: memref<?xf32>) -> (f32)
    attributes {input_segments = array<i32: 5, 0, 1>,
                result_segments = array<i32: 1, 0, 0>} {
  %index, %phase = dataflow.stream %lower, %upper, %step
      step add while slt : i64
  %sum = arith.addi %lhs, %rhs : i32
  %wide = arith.extui %sum : i32 to i64
  %addr = arith.index_cast %wide : i64 to index
  %data, %done = dataflow.load %memory[%addr] %start : memref<?xf32>
  dataflow.graph.return %done, %data : none, f32
}
