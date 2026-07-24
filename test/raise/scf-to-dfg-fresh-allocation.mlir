// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-graph-memory %t.dir/frontier.mlir -o %t.frontier.mlir
// RUN: FileCheck %s --check-prefix=FRONTIER < %t.frontier.mlir
// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/nested.mlir 2>&1 | FileCheck %s --check-prefix=NESTED

// A fresh memref.alloc result is the canonical invocation-local memory root of
// docs/spec-compiler-part-3-mem.md, and the finalized graph keeps it. In the
// graph frontier the allocation already stands at its final position, so
// preserving it there is the whole lowering action and the pass leaves it in
// place. The same allocation inside structured control is created once per
// execution of that container, no lowering reproduces that identity at the
// frontier, and it is rejected before the pass mutates the graph.

// FRONTIER-LABEL: dataflow.graph private @frontier_fresh_allocation
// FRONTIER: %[[SLOT:.*]] = memref.alloc() : memref<1xi32>
// FRONTIER: dataflow.store %[[SLOT]]
// FRONTIER: dataflow.graph.return

//--- frontier.mlir
dataflow.graph private @frontier_fresh_allocation(
    %start: none, %value: i32) -> (memref<1xi32>)
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 1>} {
  %slot = memref.alloc() : memref<1xi32>
  %index = dataflow.constant %start {const_value = 0 : index} : index
  %done = dataflow.store %slot[%index] %value %start : memref<1xi32>
  dataflow.graph.return values() streams()
      memories(%slot : memref<1xi32>) complete(%done : none)
}

// NESTED: error: loom-lower-graph-memory: operation 'memref.alloc' is not a registered canonical Dataflow actor or a supported graph-lowering operation
// NESTED-LABEL: dataflow.graph private @nested_fresh_allocation
// NESTED: scf.if
// NESTED: memref.alloc
// NESTED: memref.store
// NESTED-NOT: dataflow.demux

//--- nested.mlir
dataflow.graph private @nested_fresh_allocation(
    %start: none, %cond: i1, %value: i32) -> ()
    attributes {input_segments = array<i32: 2, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  %index = arith.constant 0 : index
  scf.if %cond {
    %slot = memref.alloc() : memref<1xi32>
    memref.store %value, %slot[%index] : memref<1xi32>
  }
  dataflow.graph.return %start : none
}
