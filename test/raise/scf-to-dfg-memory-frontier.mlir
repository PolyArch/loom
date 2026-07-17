// RUN: loom-raise-opt --split-input-file --loom-lower-graph-memory %s | FileCheck %s

memref.global "private" @frontier_a : memref<16xi32>
memref.global "private" @frontier_b : memref<16xi32>

// CHECK-LABEL: dataflow.graph.func private @frontier_straight
// CHECK-DAG: %[[A_READ:.*]] = memref.get_global @frontier_a : memref<16xi32>
// CHECK-DAG: %[[A_WRITE:.*]] = memref.get_global @frontier_a : memref<16xi32>
// CHECK-DAG: %[[B:.*]] = memref.get_global @frontier_b : memref<16xi32>
// CHECK: %[[R0:.*]], %[[D0:.*]] = dataflow.load %[[A_READ]][%arg1] %arg0 : memref<16xi32>
// CHECK: %[[R1:.*]], %[[D1:.*]] = dataflow.load %[[A_READ]][%arg2] %arg0 : memref<16xi32>
// CHECK: %[[WRITE:.*]] = dataflow.store %[[A_WRITE]][%arg1] %arg3 [[READS:%[^# ]+]]#0 : memref<16xi32>
// CHECK: %[[R2:.*]], %[[D2:.*]] = dataflow.load %[[A_READ]][%arg2] %[[WRITE]] : memref<16xi32>
// CHECK: [[READS]]:2 = dataflow.sync %[[D0]], %[[D1]] : (none, none) -> (none, none)
// CHECK: %[[RB:.*]], %[[DB:.*]] = dataflow.load %[[B]][%arg1] %arg0 : memref<16xi32>
// CHECK: dataflow.graph.return %arg0 : none
dataflow.graph.func private @frontier_straight(
    %start: none, %i: index, %j: index, %value: i32) -> none {
  %a_read = memref.get_global @frontier_a : memref<16xi32>
  %a_write = memref.get_global @frontier_a : memref<16xi32>
  %b = memref.get_global @frontier_b : memref<16xi32>
  %r0, %read0_done = dataflow.load %a_read[%i] %start : memref<16xi32>
  %r1, %read1_done = dataflow.load %a_read[%j] %start : memref<16xi32>
  %write_done = dataflow.store %a_write[%i] %value %start : memref<16xi32>
  %r2, %read2_done = dataflow.load %a_read[%j] %start : memref<16xi32>
  %rb = memref.load %b[%i] : memref<16xi32>
  dataflow.graph.return %start : none
}

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_boundary_args_may_alias
// CHECK: %[[BOUNDARY_WRITE:.*]] = dataflow.store %arg1[%arg3] %arg4 %arg0 : memref<?xi32>
// CHECK: dataflow.load %arg2[%arg3] %[[BOUNDARY_WRITE]] : memref<?xi32>
dataflow.graph.func private @frontier_boundary_args_may_alias(
    %start: none, %a: memref<?xi32>, %b: memref<?xi32>,
    %index: index, %value: i32) -> none {
  memref.store %value, %a[%index] : memref<?xi32>
  %loaded = memref.load %b[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_unknown
// CHECK: %[[UNKNOWN:.*]] = builtin.unrealized_conversion_cast %arg1, %arg2 : memref<?xi32>, memref<?xi32> to memref<?xi32>
// CHECK: %[[RA:.*]], %[[DA:.*]] = dataflow.load %arg1[%arg3] %arg0 : memref<?xi32>
// CHECK: %[[RB:.*]], %[[DB:.*]] = dataflow.load %arg2[%arg3] %arg0 : memref<?xi32>
// CHECK: %[[READS:.*]]:2 = dataflow.sync %[[DA]], %[[DB]] : (none, none) -> (none, none)
// CHECK: %[[WRITE:.*]] = dataflow.store %[[UNKNOWN]][%arg3] %arg4 %[[READS]]#0 : memref<?xi32>
// CHECK: dataflow.load %arg1[%arg3] %[[WRITE]] : memref<?xi32>
dataflow.graph.func private @frontier_unknown(
    %start: none, %a: memref<?xi32>, %b: memref<?xi32>,
    %index: index, %value: i32) -> none {
  %unknown = builtin.unrealized_conversion_cast %a, %b
      : memref<?xi32>, memref<?xi32> to memref<?xi32>
  %ra = memref.load %a[%index] : memref<?xi32>
  %rb = memref.load %b[%index] : memref<?xi32>
  memref.store %value, %unknown[%index] : memref<?xi32>
  %after = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_if_identity
// CHECK: %[[E:.*]]:2 = dataflow.demux %arg3, %arg0 : (i1, none) -> (none, none)
// CHECK: %[[W:.*]]:2 = dataflow.demux %arg3, %arg0 : (i1, none) -> (none, none)
// CHECK: %[[R:.*]]:2 = dataflow.demux %arg3, %arg0 : (i1, none) -> (none, none)
// CHECK: %[[VALUE:.*]]:2 = dataflow.demux %arg3, %arg5 : (i1, i32) -> (i32, i32)
// CHECK: %[[INDEX:.*]]:2 = dataflow.demux %arg3, %arg4 : (i1, index) -> (index, index)
// CHECK: %[[TRUE_CTRL:.*]]:2 = dataflow.sync %[[E]]#1, %[[R]]#1 : (none, none) -> (none, none)
// CHECK: %[[STORE_DONE:.*]] = dataflow.store %arg1[%[[INDEX]]#1] %[[VALUE]]#1 %[[TRUE_CTRL]]#0 : memref<?xi32>
// CHECK: %[[W_OUT:.*]] = dataflow.mux %arg3, %[[W]]#0, %[[STORE_DONE]] : (i1, none, none) -> none
// CHECK: %[[R_OUT:.*]] = dataflow.mux %arg3, %[[R]]#0, %[[STORE_DONE]] : (i1, none, none) -> none
// CHECK: %[[E_OUT:.*]] = dataflow.mux %arg3, %[[E]]#0, %[[E]]#1 : (i1, none, none) -> none
// CHECK: %[[AFTER_CTRL:.*]]:2 = dataflow.sync %[[E_OUT]], %[[W_OUT]] : (none, none) -> (none, none)
// CHECK: dataflow.load %arg1[%arg4] %[[AFTER_CTRL]]#0 : memref<?xi32>
// CHECK-NOT: arith.select
// CHECK: dataflow.graph.return %arg0 : none
dataflow.graph.func private @frontier_if_identity(
    %start: none, %a: memref<?xi32>, %b: memref<?xi32>,
    %cond: i1, %index: index, %value: i32) -> none {
  scf.if %cond {
    memref.store %value, %a[%index] : memref<?xi32>
  }
  %after = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_if_values
// CHECK: %[[THEN_VALUE:.*]]:2 = dataflow.demux %arg3, %arg4 : (i1, i32) -> (i32, i32)
// CHECK: %[[ELSE_VALUE:.*]]:2 = dataflow.demux %arg3, %arg5 : (i1, i32) -> (i32, i32)
// CHECK: %[[RESULT:.*]] = dataflow.mux %arg3, %[[ELSE_VALUE]]#0, %[[THEN_VALUE]]#1 : (i1, i32, i32) -> i32
// CHECK: dataflow.store %arg2[%arg6] %[[RESULT]]
dataflow.graph.func private @frontier_if_values(
    %start: none, %a: memref<?xi32>, %b: memref<?xi32>, %cond: i1,
    %then_value: i32, %else_value: i32, %index: index) -> none {
  %selected = scf.if %cond -> (i32) {
    memref.store %then_value, %a[%index] : memref<?xi32>
    scf.yield %then_value : i32
  } else {
    scf.yield %else_value : i32
  }
  memref.store %selected, %b[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_for
// CHECK: %[[IV:.*]], %[[PHASE:.*]] = dataflow.stream %arg3, %arg4, %arg5 step add while slt : i64
// CHECK: %[[EXEC_RAW:.*]] = dataflow.carry %[[PHASE]], %arg0,
// CHECK: %[[EXEC_LANES:.*]]:2 = dataflow.demux %[[PHASE]], %[[EXEC_RAW]] : (i1, none) -> (none, none)
// CHECK: %[[VALUE_RAW:.*]] = dataflow.invariant %[[PHASE]], %arg7 : i32
// CHECK: %{{.*}}, %[[BODY_VALUE:.*]] = dataflow.gate %[[PHASE]], %[[VALUE_RAW]] : i32
// CHECK: %[[W_RAW:.*]] = dataflow.carry %[[PHASE]], %arg0,
// CHECK: %[[R_RAW:.*]] = dataflow.carry %[[PHASE]], %arg0,
// CHECK: %[[W_LANES:.*]]:2 = dataflow.demux %[[PHASE]], %[[W_RAW]] : (i1, none) -> (none, none)
// CHECK: %[[R_LANES:.*]]:2 = dataflow.demux %[[PHASE]], %[[R_RAW]] : (i1, none) -> (none, none)
// CHECK: dataflow.load %arg1[{{.*}}]
// CHECK: %[[STORE_DONE:.*]] = dataflow.store %arg1[{{.*}}] %[[BODY_VALUE]]
// CHECK: dataflow.load %arg1[%arg6]
// CHECK-NOT: scf.for
dataflow.graph.func private @frontier_for(
    %start: none, %a: memref<?xi32>, %b: memref<?xi32>,
    %lb: i64, %ub: i64, %step: i64, %after_index: index,
    %value: i32) -> none {
  scf.for %i = %lb to %ub step %step : i64 {
    %index = arith.index_cast %i : i64 to index
    %loaded = memref.load %a[%index] : memref<?xi32>
    memref.store %value, %a[%index] : memref<?xi32>
  }
  %after = memref.load %a[%after_index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_for_zero_trip
// CHECK: %[[ZERO_IV:.*]], %[[ZERO_PHASE:.*]] = dataflow.stream %arg2, %arg2, %arg3 step add while slt : i64
// CHECK: %[[ZERO_EXEC_RAW:.*]] = dataflow.carry %[[ZERO_PHASE]], %arg0,
// CHECK: %[[ZERO_EXEC_LANES:.*]]:2 = dataflow.demux %[[ZERO_PHASE]], %[[ZERO_EXEC_RAW]] : (i1, none) -> (none, none)
// CHECK: %[[ZERO_VALUE_RAW:.*]] = dataflow.carry %[[ZERO_PHASE]], %arg5,
// CHECK: %[[ZERO_VALUE_LANES:.*]]:2 = dataflow.demux %[[ZERO_PHASE]], %[[ZERO_VALUE_RAW]] : (i1, i32) -> (i32, i32)
// CHECK: %[[ZERO_W_RAW:.*]] = dataflow.carry %[[ZERO_PHASE]], %arg0,
// CHECK: %[[ZERO_W_LANES:.*]]:2 = dataflow.demux %[[ZERO_PHASE]], %[[ZERO_W_RAW]] : (i1, none) -> (none, none)
// CHECK: %[[ZERO_AFTER_CTRL:.*]]:2 = dataflow.sync %[[ZERO_EXEC_LANES]]#0, %[[ZERO_W_LANES]]#0 : (none, none) -> (none, none)
// CHECK: dataflow.load %arg1[%arg4] %[[ZERO_AFTER_CTRL]]#0 : memref<?xi32>
// CHECK: dataflow.graph.return %arg0, %[[ZERO_VALUE_LANES]]#0 : none, i32
dataflow.graph.func private @frontier_for_zero_trip(
    %start: none, %a: memref<?xi32>, %bound: i64, %step: i64,
    %index: index, %value: i32) -> (none, i32) {
  %result = scf.for %i = %bound to %bound step %step
      iter_args(%state = %value) -> (i32) : i64 {
    memref.store %state, %a[%index] : memref<?xi32>
    scf.yield %state : i32
  }
  %after = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start, %result : none, i32
}

// CHECK-LABEL: dataflow.graph.func private @frontier_for_values
// CHECK: %[[VALUE_RAW:.*]] = dataflow.carry %[[VALUE_PHASE:.*]], %arg6,
// CHECK: %[[VALUE_LANES:.*]]:2 = dataflow.demux %[[VALUE_PHASE]], %[[VALUE_RAW]] : (i1, i32) -> (i32, i32)
// CHECK: %[[NEXT:.*]] = arith.addi %[[VALUE_LANES]]#1,
// CHECK: dataflow.store %arg1[%arg5] %[[VALUE_LANES]]#0
// CHECK-NOT: scf.for
dataflow.graph.func private @frontier_for_values(
    %start: none, %a: memref<?xi32>, %lb: i64, %ub: i64, %step: i64,
    %index: index, %init: i32, %increment: i32) -> none {
  %result = scf.for %i = %lb to %ub step %step
      iter_args(%value = %init) -> (i32) : i64 {
    %next = arith.addi %value, %increment : i32
    scf.yield %next : i32
  }
  memref.store %result, %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_for_descending
// CHECK: %[[DESC_IV:.*]], %[[DESC_PHASE:.*]] = dataflow.stream %arg2, %arg3, %arg4 step add while sgt : i64
// CHECK: %[[DESC_VALUE:.*]] = dataflow.carry %[[DESC_PHASE]], %arg5,
// CHECK: %[[DESC_LANES:.*]]:2 = dataflow.demux %[[DESC_PHASE]], %[[DESC_VALUE]] : (i1, i64) -> (i64, i64)
// CHECK: dataflow.store %arg1[%arg6] %[[DESC_LANES]]#0
// CHECK-NOT: scf.for
dataflow.graph.func private @frontier_for_descending(
    %start: none, %a: memref<?xi64>, %lb: i64, %ub: i64, %step: i64,
    %init: i64, %index: index) -> none {
  %result = scf.for %i = %lb to %ub step %step
      iter_args(%value = %init) -> (i64) : i64 {
    %next = arith.addi %value, %step : i64
    scf.yield %next : i64
  } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 4 : i64}
  memref.store %result, %a[%index] : memref<?xi64>
  dataflow.graph.return %start : none
}

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_while_final_false
// CHECK: %[[EXEC_RAW:.*]] = dataflow.carry %[[COND:.*]], %arg0,
// CHECK: %[[W_RAW:.*]] = dataflow.carry %[[COND]], %arg0,
// CHECK: %[[R_RAW:.*]] = dataflow.carry %[[COND]], %arg0,
// CHECK: %[[BEFORE_LOAD:.*]], %[[BEFORE_DONE:.*]] = dataflow.load %arg1[{{.*}}]
// CHECK: %[[R_BEFORE:.*]]:2 = dataflow.sync %{{.*}}, %[[BEFORE_DONE]] : (none, none) -> (none, none)
// CHECK: %[[EXEC_EXIT:.*]]:2 = dataflow.demux %[[COND]], {{.*}} : (i1, none) -> (none, none)
// CHECK: %{{.*}}, %[[AFTER_EXEC:.*]] = dataflow.gate %[[COND]], {{.*}} : none
// CHECK: %[[R_LANES:.*]]:2 = dataflow.demux %[[COND]], %[[R_BEFORE]]#0 : (i1, none) -> (none, none)
// CHECK: %[[POST_CTRL:.*]]:2 = dataflow.sync %{{.*}}, %[[R_LANES]]#0 : (none, none) -> (none, none)
// CHECK: dataflow.store %arg1[%arg5] %{{.*}} %[[POST_CTRL]]#0 : memref<?xi32>
// CHECK-NOT: scf.while
dataflow.graph.func private @frontier_while_final_false(
    %start: none, %a: memref<?xi32>, %init: i64, %limit: i64,
    %one: i64, %post_index: index, %post_value: i32) -> none {
  %result = scf.while (%i = %init) : (i64) -> i64 {
    %index = arith.index_cast %i : i64 to index
    %loaded = memref.load %a[%index] : memref<?xi32>
    %continue = arith.cmpi slt, %i, %limit : i64
    scf.condition(%continue) %i : i64
  } do {
  ^bb0(%after_i: i64):
    %next = arith.addi %after_i, %one : i64
    scf.yield %next : i64
  }
  memref.store %post_value, %a[%post_index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_while_carried_condition
// CHECK: %[[CARRIED_SELECTOR:.*]] = dataflow.carry {{%.*}}, %arg2,
// CHECK: dataflow.demux %[[CARRIED_SELECTOR]], {{.*}} : (i1, none) -> (none, none)
// CHECK: dataflow.gate %[[CARRIED_SELECTOR]], {{.*}} : none
// CHECK: dataflow.demux %[[CARRIED_SELECTOR]], {{.*}} : (i1, i32) -> (i32, i32)
// CHECK-NOT: scf.while
dataflow.graph.func private @frontier_while_carried_condition(
    %start: none, %a: memref<?xi32>, %initial_condition: i1,
    %next_condition: i1, %initial_value: i32, %index: index) -> none {
  %result:2 = scf.while (%condition = %initial_condition,
                         %value = %initial_value) : (i1, i32) -> (i1, i32) {
    scf.condition(%condition) %condition, %value : i1, i32
  } do {
  ^bb0(%condition: i1, %value: i32):
    scf.yield %next_condition, %value : i1, i32
  }
  memref.store %result#1, %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_while_captured_condition
// CHECK: %[[CAPTURED_SELECTOR:.*]] = dataflow.invariant {{%.*}}, %arg2 : i1
// CHECK: dataflow.demux %[[CAPTURED_SELECTOR]], {{.*}} : (i1, none) -> (none, none)
// CHECK: dataflow.gate %[[CAPTURED_SELECTOR]], {{.*}} : none
// CHECK: dataflow.demux %[[CAPTURED_SELECTOR]], {{.*}} : (i1, i32) -> (i32, i32)
// CHECK-NOT: scf.while
dataflow.graph.func private @frontier_while_captured_condition(
    %start: none, %a: memref<?xi32>, %captured_condition: i1,
    %initial_value: i32, %index: index) -> none {
  %result = scf.while (%value = %initial_value) : (i32) -> i32 {
    scf.condition(%captured_condition) %value : i32
  } do {
  ^bb0(%value: i32):
    scf.yield %value : i32
  }
  memref.store %result, %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_while_nested_if_condition
// CHECK: %[[NESTED_SELECTOR:.*]] = dataflow.mux %{{.*}}, {{.*}} : (i1, i1, i1) -> i1
// CHECK: dataflow.demux %[[NESTED_SELECTOR]], {{.*}} : (i1, none) -> (none, none)
// CHECK: dataflow.gate %[[NESTED_SELECTOR]], {{.*}} : none
// CHECK: dataflow.demux %[[NESTED_SELECTOR]], {{.*}} : (i1, i32) -> (i32, i32)
// CHECK-NOT: scf.if
// CHECK-NOT: scf.while
dataflow.graph.func private @frontier_while_nested_if_condition(
    %start: none, %a: memref<?xi32>, %guard: i1, %true_condition: i1,
    %false_condition: i1, %initial_value: i32, %index: index) -> none {
  %result = scf.while (%value = %initial_value) : (i32) -> i32 {
    %condition = scf.if %guard -> (i1) {
      scf.yield %true_condition : i1
    } else {
      scf.yield %false_condition : i1
    }
    scf.condition(%condition) %value : i32
  } do {
  ^bb0(%value: i32):
    scf.yield %value : i32
  }
  memref.store %result, %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_nested_for_while
// CHECK: dataflow.stream
// CHECK: dataflow.carry
// CHECK: dataflow.carry
// CHECK: dataflow.load
// CHECK-NOT: scf.for
// CHECK-NOT: scf.while
dataflow.graph.func private @frontier_nested_for_while(
    %start: none, %a: memref<?xi32>, %lb: i64, %ub: i64, %step: i64,
    %limit: i64, %one: i64) -> none {
  scf.for %outer = %lb to %ub step %step : i64 {
    %result = scf.while (%inner = %outer) : (i64) -> i64 {
      %index = arith.index_cast %inner : i64 to index
      %loaded = memref.load %a[%index] : memref<?xi32>
      %continue = arith.cmpi slt, %inner, %limit : i64
      scf.condition(%continue) %inner : i64
    } do {
    ^bb0(%after: i64):
      %next = arith.addi %after, %one : i64
      scf.yield %next : i64
    }
  }
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_nested_if_for
// CHECK: dataflow.demux %arg3, %arg0
// CHECK: dataflow.stream
// CHECK: dataflow.mux %arg3
// CHECK-NOT: scf.if
// CHECK-NOT: scf.for
dataflow.graph.func private @frontier_nested_if_for(
    %start: none, %a: memref<?xi32>, %b: memref<?xi32>, %cond: i1,
    %lb: i64, %ub: i64, %step: i64, %value: i32) -> none {
  scf.if %cond {
    scf.for %i = %lb to %ub step %step : i64 {
      %index = arith.index_cast %i : i64 to index
      memref.store %value, %a[%index] : memref<?xi32>
    }
  }
  dataflow.graph.return %start : none
}

// CHECK-LABEL: dataflow.graph.func private @frontier_nested_for_if
// CHECK: dataflow.stream
// CHECK: dataflow.demux %{{.*}}, %{{.*}} : (i1, none) -> (none, none)
// CHECK: dataflow.mux
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
dataflow.graph.func private @frontier_nested_for_if(
    %start: none, %a: memref<?xi32>, %lb: i64, %ub: i64, %step: i64,
    %limit: i64, %index: index, %value: i32) -> none {
  scf.for %i = %lb to %ub step %step : i64 {
    %condition = arith.cmpi slt, %i, %limit : i64
    scf.if %condition {
      memref.store %value, %a[%index] : memref<?xi32>
    }
  }
  dataflow.graph.return %start : none
}
