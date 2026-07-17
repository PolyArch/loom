// RUN: not loom-raise-opt --loom-lower-graph-memory %s 2>&1 | FileCheck %s

// Memory capabilities cannot become dynamic dataflow carry state. Until a
// source pointer recurrence has been projected to an explicit index domain,
// graph-region lowering must fail without partially rewriting the graph.
// CHECK: error: cannot lower loop-carried memory capability '!llvm.ptr' through dataflow.carry
// CHECK-NOT: dataflow.carry
dataflow.graph.func private @pointer_carry(
    %start: none, %lb: index, %ub: index, %step: index, %pointer: !llvm.ptr)
    -> (none, !llvm.ptr)
    attributes {input_segments = array<i32: 3, 0, 1>,
                result_segments = array<i32: 0, 0, 1>} {
  %result = scf.for %i = %lb to %ub step %step
      iter_args(%current = %pointer) -> (!llvm.ptr) {
    %next = llvm.getelementptr %current[1]
        : (!llvm.ptr) -> !llvm.ptr, i8
    scf.yield %next : !llvm.ptr
  }
  dataflow.graph.return values() streams() memories(%result : !llvm.ptr)
      complete(%start : none)
}
