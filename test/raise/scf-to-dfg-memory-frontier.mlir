// RUN: loom-raise-opt --split-input-file --loom-lower-graph-memory %s | FileCheck %s

// CHECK-LABEL: dataflow.graph.func private @frontier_straight
// CHECK: %[[R0:.*]], %[[D0:.*]] = dataflow.load %arg1[%arg3] %arg0 : memref<?xi32>
// CHECK: %[[R1:.*]], %[[D1:.*]] = dataflow.load %arg1[%arg4] %arg0 : memref<?xi32>
// CHECK: %[[WRITE:.*]] = dataflow.store %arg1[%arg3] %arg5 [[READS:%[^# ]+]]#0 : memref<?xi32>
// CHECK: %[[R2:.*]], %[[D2:.*]] = dataflow.load %arg1[%arg4] %[[WRITE]] : memref<?xi32>
// CHECK: [[READS]]:2 = dataflow.sync %[[D0]], %[[D1]] : (none, none) -> (none, none)
// CHECK: %[[RB:.*]], %[[DB:.*]] = dataflow.load %arg2[%arg3] %arg0 : memref<?xi32>
// CHECK: dataflow.graph.return %arg0 : none
dataflow.graph.func private @frontier_straight(
    %start: none, %a: memref<?xi32>, %b: memref<?xi32>,
    %i: index, %j: index, %value: i32) -> none {
  %r0, %read0_done = dataflow.load %a[%i] %start : memref<?xi32>
  %r1, %read1_done = dataflow.load %a[%j] %start : memref<?xi32>
  %write_done = dataflow.store %a[%i] %value %start : memref<?xi32>
  %r2, %read2_done = dataflow.load %a[%j] %start : memref<?xi32>
  %rb = memref.load %b[%i] : memref<?xi32>
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
// CHECK: %[[W_RAW:.*]] = dataflow.carry %[[PHASE]], %arg0,
// CHECK: %[[R_RAW:.*]] = dataflow.carry %[[PHASE]], %arg0,
// CHECK: %[[W_LANES:.*]]:2 = dataflow.demux %[[PHASE]], %[[W_RAW]] : (i1, none) -> (none, none)
// CHECK: %[[R_LANES:.*]]:2 = dataflow.demux %[[PHASE]], %[[R_RAW]] : (i1, none) -> (none, none)
// CHECK: dataflow.load %arg1[{{.*}}]
// CHECK: %[[STORE_DONE:.*]] = dataflow.store %arg1[{{.*}}]
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
// CHECK: %[[ZERO_W_RAW:.*]] = dataflow.carry %[[ZERO_PHASE]], %arg0,
// CHECK: %[[ZERO_W_LANES:.*]]:2 = dataflow.demux %[[ZERO_PHASE]], %[[ZERO_W_RAW]] : (i1, none) -> (none, none)
// CHECK: dataflow.graph.return %arg0 : none
dataflow.graph.func private @frontier_for_zero_trip(
    %start: none, %a: memref<?xi32>, %bound: i64, %step: i64,
    %index: index, %value: i32) -> none {
  scf.for %i = %bound to %bound step %step : i64 {
    memref.store %value, %a[%index] : memref<?xi32>
  }
  dataflow.graph.return %start : none
}

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_while_final_false
// CHECK: %[[EXEC_RAW:.*]] = dataflow.carry %[[COND:.*]], %arg0,
// CHECK: %[[W_RAW:.*]] = dataflow.carry %[[COND]], %arg0,
// CHECK: %[[R_RAW:.*]] = dataflow.carry %[[COND]], %arg0,
// CHECK: %[[BEFORE_LOAD:.*]], %[[BEFORE_DONE:.*]] = dataflow.load %arg1[{{.*}}]
// CHECK: %[[R_BEFORE:.*]]:2 = dataflow.sync %{{.*}}, %[[BEFORE_DONE]] : (none, none) -> (none, none)
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

// -----

// CHECK-LABEL: dataflow.graph.func private @frontier_return_is_not_completion
// CHECK: %[[DONE:.*]] = dataflow.store %arg1[%arg2] %arg3 %arg0 : memref<?xi32>
// CHECK: dataflow.graph.return %arg0 : none
// CHECK-NOT: dataflow.graph.return %[[DONE]]
dataflow.graph.func private @frontier_return_is_not_completion(
    %start: none, %a: memref<?xi32>, %index: index, %value: i32) -> none {
  memref.store %value, %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}
