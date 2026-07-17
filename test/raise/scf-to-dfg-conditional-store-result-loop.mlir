// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s --implicit-check-not=loom.conditional_store_

// Compact-style loops carry a write cursor through a result-bearing
// conditional store. The store branch must be routed by the condition, and
// skipped-store lanes must still produce one done token per loop item so the
// graph rendezvous does not silently drop iterations.

// CHECK-LABEL: dataflow.graph private @g_conditional_store_result_loop
// CHECK: %[[IDX:.*]], %[[RWC:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[CURSOR_RAW:.*]] = dataflow.carry %[[RWC]], %arg6,
// CHECK: %[[CURSOR_EXIT:.*]]:2 = dataflow.demux %[[RWC]], %[[CURSOR_RAW]] : (i1, i32) -> (i32, i32)
// CHECK: %[[DATA:.*]], %{{.*}} = dataflow.load
// CHECK: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[DATA]], %{{.*}} : i32
// CHECK: %[[STORE_CTRL:.*]]:2 = dataflow.demux %[[IS_ZERO]], {{.*}} : (i1, none) -> (none, none)
// CHECK: %[[STORE_CURSOR:.*]]:2 = dataflow.demux %[[IS_ZERO]], %[[CURSOR_EXIT]]#1 : (i1, i32) -> (i32, i32)
// CHECK: %[[STORE_DATA:.*]]:2 = dataflow.demux %[[IS_ZERO]], %[[DATA]] : (i1, i32) -> (i32, i32)
// CHECK: %[[STORE_DONE:.*]] = dataflow.store {{.*}} %[[STORE_DATA]]#0 {{.*}}
// CHECK: %[[NEXT:.*]] = arith.addi %[[STORE_CURSOR]]#0, %{{.*}} : i32
// CHECK: dataflow.mux %[[IS_ZERO]], %[[NEXT]], %[[STORE_CURSOR]]#1 : (i1, i32, i32) -> i32
// CHECK: dataflow.mux %[[IS_ZERO]], %[[STORE_DONE]], {{.*}} : (i1, none, none) -> none
// CHECK: dataflow.sync {{.*}}, %[[CURSOR_EXIT]]#0 : (none, i32) -> (none, i32)
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: dataflow.graph.return
dataflow.graph private @g_conditional_store_result_loop(
    %ctrl: none, %lb: i64, %ub: i64, %step: i64, %zero: i32,
    %one: i32, %init: i32, %input: !llvm.ptr, %output: !llvm.ptr)
    -> (i32)
    attributes {input_segments = array<i32: 6, 0, 2>,
                result_segments = array<i32: 1, 0, 0>} {
  %r = scf.for %i = %lb to %ub step %step iter_args(%cursor = %init)
      -> (i32) : i64 {
    %in_mem = builtin.unrealized_conversion_cast %input
        : !llvm.ptr to memref<?xi32>
    %in_idx = arith.index_cast %i : i64 to index
    %data, %done = dataflow.load %in_mem[%in_idx] %ctrl : memref<?xi32>
    %is_zero = arith.cmpi eq, %data, %zero : i32
    %next = scf.if %is_zero -> (i32) {
      scf.yield %cursor : i32
    } else {
      %cursor64 = llvm.zext %cursor : i32 to i64
      %out_mem = builtin.unrealized_conversion_cast %output
          : !llvm.ptr to memref<?xi32>
      %out_idx_pre = arith.index_cast %cursor64 : i64 to index
      %store = dataflow.store %out_mem[%out_idx_pre] %data %ctrl
          : memref<?xi32>
      %inc = arith.addi %cursor, %one : i32
      scf.yield %inc : i32
    }
    scf.yield %next : i32
  } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 2 : i64}
  dataflow.graph.return %ctrl, %r : none, i32
}

// CHECK-LABEL: dataflow.graph private @g_conditional_store_result_then_loop
// CHECK: %[[IDX2:.*]], %[[RWC2:.*]] = dataflow.stream %arg1, %arg2, %arg3
// CHECK: %[[CURSOR_RAW2:.*]] = dataflow.carry %[[RWC2]], %arg6,
// CHECK: %[[CURSOR_EXIT2:.*]]:2 = dataflow.demux %[[RWC2]], %[[CURSOR_RAW2]] : (i1, i32) -> (i32, i32)
// CHECK: %[[DATA2:.*]], %{{.*}} = dataflow.load
// CHECK: %[[IS_ZERO2:.*]] = arith.cmpi eq, %[[DATA2]], %{{.*}} : i32
// CHECK: %[[STORE_CTRL2:.*]]:2 = dataflow.demux %[[IS_ZERO2]], {{.*}} : (i1, none) -> (none, none)
// CHECK: %[[STORE_CURSOR2:.*]]:2 = dataflow.demux %[[IS_ZERO2]], %[[CURSOR_EXIT2]]#1 : (i1, i32) -> (i32, i32)
// CHECK: %[[STORE_DATA2:.*]]:2 = dataflow.demux %[[IS_ZERO2]], %[[DATA2]] : (i1, i32) -> (i32, i32)
// CHECK: %[[STORE_DONE2:.*]] = dataflow.store {{.*}} %[[STORE_DATA2]]#1 {{.*}}
// CHECK: %[[NEXT2:.*]] = arith.addi %[[STORE_CURSOR2]]#1, %{{.*}} : i32
// CHECK: dataflow.mux %[[IS_ZERO2]], %[[STORE_CURSOR2]]#0, %[[NEXT2]] : (i1, i32, i32) -> i32
// CHECK: dataflow.mux %[[IS_ZERO2]], {{.*}}, %[[STORE_DONE2]] : (i1, none, none) -> none
// CHECK: dataflow.sync {{.*}}, %[[CURSOR_EXIT2]]#0 : (none, i32) -> (none, i32)
// CHECK-NOT: scf.for
// CHECK-NOT: scf.if
// CHECK: dataflow.graph.return
dataflow.graph private @g_conditional_store_result_then_loop(
    %ctrl: none, %lb: i64, %ub: i64, %step: i64, %zero: i32,
    %one: i32, %init: i32, %input: !llvm.ptr, %output: !llvm.ptr)
    -> (i32)
    attributes {input_segments = array<i32: 6, 0, 2>,
                result_segments = array<i32: 1, 0, 0>} {
  %r = scf.for %i = %lb to %ub step %step iter_args(%cursor = %init)
      -> (i32) : i64 {
    %in_mem = builtin.unrealized_conversion_cast %input
        : !llvm.ptr to memref<?xi32>
    %in_idx = arith.index_cast %i : i64 to index
    %data, %done = dataflow.load %in_mem[%in_idx] %ctrl : memref<?xi32>
    %is_zero = arith.cmpi eq, %data, %zero : i32
    %next = scf.if %is_zero -> (i32) {
      %cursor64 = llvm.zext %cursor : i32 to i64
      %out_mem = builtin.unrealized_conversion_cast %output
          : !llvm.ptr to memref<?xi32>
      %out_idx_pre = arith.index_cast %cursor64 : i64 to index
      %store = dataflow.store %out_mem[%out_idx_pre] %data %ctrl
          : memref<?xi32>
      %inc = arith.addi %cursor, %one : i32
      scf.yield %inc : i32
    } else {
      scf.yield %cursor : i32
    }
    scf.yield %next : i32
  } {loom.stream_step_kind = 0 : i32, loom.stream_predicate = 2 : i64}
  dataflow.graph.return %ctrl, %r : none, i32
}
