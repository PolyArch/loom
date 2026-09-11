// RUN: loom-raise-opt --loom-lower-scf-to-dfg %s | FileCheck %s

// Execution permission and the memory components are distinct token systems.
// A selection inside a loop body projects each of them into lanes, and it must
// project them separately even when they carry the same event at loop entry.
// One actor carrying both roles puts the execution recurrence on the memory
// frontier's own alignment path: a loop whose iterations are proven
// independent counts completions on a second stream, and that stream's carry
// feedback then reaches the shared lane projection, walks through the
// execution recurrence, and arrives back at the same projection. The witness
// stops being provably one-shot and publication rejects the graph.
//
// The guarded region owns a nested accumulating loop so the selection's true
// lane advances execution, which is the shape a source loop nest with a
// guarded reduction produces.

// CHECK-LABEL: dataflow.graph private @independent_selected_memory
// CHECK: %{{.*}}, %[[PHASE:.*]] = dataflow.stream
// CHECK: %[[GUARD_RAW:.*]] = dataflow.invariant %[[PHASE]], %arg3 : i1
// CHECK: %[[GUARD:.*]]:2 = dataflow.demux %[[PHASE]], %[[GUARD_RAW]] : (i1, i1) -> (i1, i1)
// CHECK: %[[LANE_EXECUTION:.*]]:2 = dataflow.demux %[[GUARD]]#1, %{{.*}} : (i1, none) -> (none, none)
// CHECK: %[[LANE_MEMORY:.*]]:2 = dataflow.demux %[[GUARD]]#1, %{{.*}} : (i1, none) -> (none, none)
// CHECK: dataflow.graph.return

dataflow.graph private @independent_selected_memory(
    %start: none, %outer: index, %inner: index, %guard: i1,
    %source: memref<8xf32> {llvm.noalias},
    %target: memref<8xf32> {llvm.noalias}) -> ()
    attributes {input_segments = array<i32: 3, 0, 2>,
                result_segments = array<i32: 0, 0, 0>} {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %init = arith.constant 0.000000e+00 : f32
  scf.for %i = %zero to %outer step %one {
    %sum = scf.if %guard -> (f32) {
      %total = scf.for %j = %zero to %inner step %one
          iter_args(%value = %init) -> (f32) {
        %element = memref.load %source[%j] : memref<8xf32>
        %next = arith.addf %value, %element : f32
        scf.yield %next : f32
      }
      scf.yield %total : f32
    } else {
      scf.yield %init : f32
    }
    memref.store %sum, %target[%i] : memref<8xf32>
  }
  dataflow.graph.return %start : none
}
