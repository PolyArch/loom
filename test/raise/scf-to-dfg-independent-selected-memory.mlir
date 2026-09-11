// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// A loop whose iterations are proven independent counts completions on its own
// stream, so that stream's carry feedback must stay alignable to its own phase.
// The body's memory precondition keeps its own replay of the partition
// frontier even when that frontier is the same event as the incoming execution
// permission. One actor carrying both roles would let a selection inside the
// body project a single lane demux for both token systems, and the completion
// witness would then only be provable by walking through the execution
// recurrence and back into that same actor.

// CHECK-LABEL: dataflow.graph private @independent_selected_memory
// CHECK: %{{.*}}, %[[PHASE:.*]] = dataflow.stream
// CHECK: %[[EXEC_RAW:.*]] = dataflow.carry %[[PHASE]], %arg0,
// CHECK: %[[EXEC:.*]]:2 = dataflow.demux %[[PHASE]], %[[EXEC_RAW]] : (i1, none) -> (none, none)
// CHECK: %{{.*}}, %[[COMPLETION_PHASE:.*]] = dataflow.stream
// CHECK: %[[COMPLETION_RAW:.*]] = dataflow.carry %[[COMPLETION_PHASE]], %arg0,
// CHECK: %[[COMPLETION:.*]]:2 = dataflow.demux %[[COMPLETION_PHASE]], %[[COMPLETION_RAW]] : (i1, none) -> (none, none)
// CHECK: %[[READY:.*]] = dataflow.invariant %[[PHASE]], %arg0 : none
// CHECK: %[[READY_LANES:.*]]:2 = dataflow.demux %[[PHASE]], %[[READY]] : (i1, none) -> (none, none)
// CHECK: %[[LANE_EXEC:.*]]:2 = dataflow.demux %{{.*}}, %[[EXEC]]#1 : (i1, none) -> (none, none)
// CHECK: %[[LANE_MEMORY:.*]]:2 = dataflow.demux %{{.*}}, %[[READY_LANES]]#1 : (i1, none) -> (none, none)
// CHECK: dataflow.graph.return values() streams() memories() complete(%{{.*}}, %[[COMPLETION]]#0 : none, none)

dataflow.graph private @independent_selected_memory(
    %start: none, %upper: index, %guard: i1, %value: i32,
    %source: memref<8xi32> {llvm.noalias},
    %target: memref<8xi32> {llvm.noalias}) -> ()
    attributes {input_segments = array<i32: 3, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %upper step %one {
    %selected = scf.if %guard -> (i32) {
      %loaded = memref.load %source[%i] : memref<8xi32>
      scf.yield %loaded : i32
    } else {
      scf.yield %value : i32
    }
    memref.store %selected, %target[%i] : memref<8xi32>
  }
  dataflow.graph.return %start : none
}
