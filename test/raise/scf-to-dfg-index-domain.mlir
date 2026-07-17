// RUN: loom-raise-opt --loom-lower-graph-memory %s | FileCheck %s

// CHECK-LABEL: dataflow.graph.func private @narrow_projected_carry
// CHECK: %[[INIT:.*]] = arith.index_cast %arg4 : i64 to index
// CHECK: %[[STEP:.*]] = arith.index_cast %arg3 : i64 to index
// CHECK: %[[CARRY:.*]] = dataflow.carry %{{.*}}, %[[INIT]], %[[NEXT:.*]] : index
// CHECK: %[[EXIT:.*]]:2 = dataflow.demux %{{.*}}, %[[CARRY]] : (i1, index) -> (index, index)
// CHECK: %[[NEXT]] = arith.addi %{{.*}}, %{{.*}} : index
// CHECK: dataflow.load %arg6[%{{.*}}]
// CHECK: arith.index_cast %[[EXIT]]#0 : index to i64
dataflow.graph.func private @narrow_projected_carry(
    %start: none, %init: i64, %limit: i64, %step: i64, %cursor0: i64,
    %seed: f32, %memory: memref<?xf32>) -> (none, i64, f32)
    attributes {input_segments = array<i32: 5, 0, 1>,
                result_segments = array<i32: 2, 0, 0>} {
  %index, %phase = dataflow.stream %init, %limit, %step
      step add while slt : i64
  %step_raw = dataflow.invariant %phase, %step : i64
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

// Failed carry materialization must erase the speculative index constant and
// preserve the original i64 recurrence.
// CHECK-LABEL: dataflow.graph.func private @rollback_partial_carry
// CHECK: dataflow.constant %arg0 {const_value = 4 : i64} : i64
// CHECK-NOT: dataflow.constant %arg0 {const_value = 4 : index} : index
// CHECK: dataflow.carry %{{.*}} : i64
// CHECK-NOT: dataflow.carry %{{.*}} : index
dataflow.graph.func private @rollback_partial_carry(
    %start: none, %init: i64, %limit: i64, %step: i64,
    %memory: memref<?xf32>) -> (none, i64, f32)
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
// CHECK-LABEL: dataflow.graph.func private @rollback_partial_address
// CHECK: dataflow.constant %arg0 {const_value = 4 : i64} : i64
// CHECK-NOT: dataflow.constant %arg0 {const_value = 4 : index} : index
// CHECK-NOT: arith.addi {{.*}} : index
// CHECK: arith.index_cast %{{.*}} : i64 to index
dataflow.graph.func private @rollback_partial_address(
    %start: none, %phase: i1, %init: i64, %next: i64,
    %memory: memref<?xf32>) -> (none, f32)
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
