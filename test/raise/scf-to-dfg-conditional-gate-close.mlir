// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: FileCheck %s < %t.lowered.mlir
// RUN: loom-dfg-sim %t.lowered.mlir --graph conditional_gate_close \
// RUN:   --arg 0=2 --arg 1=7 --memref 2=0,0 --output %t.active.json
// RUN: loom-dfg-sim %t.lowered.mlir --graph conditional_gate_close \
// RUN:   --arg 0=0 --arg 1=7 --memref 2=0,0 --output %t.bypass.json

// CHECK-LABEL: dataflow.graph private @conditional_gate_close
// CHECK: %[[EMPTY:.*]] = arith.cmpi eq,
// CHECK: %[[GATE_PHASE:.*]], %[[BODY_VALUE:.*]] = dataflow.gate
// CHECK: %[[GATE_CLOSE:.*]]:2 = dataflow.demux %[[GATE_PHASE]], %[[BODY_VALUE]] : (i1, i32) -> (i32, i32)
// CHECK: dataflow.store {{.*}} %[[BODY_VALUE]]
// CHECK: %[[ACTIVE_COMPLETE:.*]]:2 = dataflow.sync {{.*}}, %[[GATE_CLOSE]]#0 : (none, i32) -> (none, i32)
// CHECK: %[[LOOP_COMPLETE:.*]] = dataflow.mux {{.*}}, {{.*}}, %[[ACTIVE_COMPLETE]]#0 : (i1, none, none) -> none
// CHECK: dataflow.mux %[[EMPTY]], %[[LOOP_COMPLETE]], {{.*}} : (i1, none, none) -> none
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
