// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph conditional_gate_close \
// RUN:   --arg 0=2 --arg 1=7 --memref 2=0,0 --output %t.active.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph conditional_gate_close \
// RUN:   --arg 0=0 --arg 1=7 --memref 2=0,0 --output %t.bypass.json

// The loop runs only on the else lane of the guard, so the capture close has
// to retire under the enclosing selection as well as under the loop. The
// close is the false lane of the capture's own projection and carries one
// token per activation, zero-trip included, so the loop exit joins it with no
// empty-loop selection of its own.
// CHECK-LABEL: dataflow.graph private @conditional_gate_close
// CHECK-NOT: dataflow.gate
// CHECK: %[[EMPTY:.*]] = arith.cmpi eq,
// CHECK: %[[VALUE_RAW:.*]] = dataflow.invariant %{{.*}} : i32
// CHECK: %[[VALUE_LANES:.*]]:2 = dataflow.demux %{{.*}}, %[[VALUE_RAW]] : (i1, i32) -> (i32, i32)
// CHECK-NOT: dataflow.gate
// CHECK: dataflow.store {{.*}} %[[VALUE_LANES]]#1
// CHECK: %[[LOOP_COMPLETE:.*]]:2 = dataflow.sync %{{.*}}, %[[VALUE_LANES]]#0 : (none, i32) -> (none, i32)
// CHECK: dataflow.mux %[[EMPTY]], %[[LOOP_COMPLETE]]#0, {{.*}} : (i1, none, none) -> none
// CHECK: dataflow.graph.return

dataflow.graph private @conditional_gate_close(
    %start: none, %count: i32, %value: i32, %buffer: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 2, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %zero_i32 = arith.constant 0 : i32
  %zero_i64 = arith.constant 0 : i64
  %one_i64 = arith.constant 1 : i64
  %empty = arith.cmpi eq, %count, %zero_i32 : i32
  %limit = arith.extui %count : i32 to i64
  scf.if %empty {
  } else {
    scf.for %index = %zero_i64 to %limit step %one_i64 : i64 {
      %memory_index = arith.index_cast %index : i64 to index
      memref.store %value, %buffer[%memory_index] : memref<?xi32>
    }
  }
  dataflow.graph.return %start : none
}
