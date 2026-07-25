// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/fence.mlir 2>&1 | FileCheck %s --check-prefix=FENCE
// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/atomic.mlir 2>&1 | FileCheck %s --check-prefix=ATOMIC

// An effectful canonical actor that no lowering capability covers is rejected
// during preflight inside a serial scf container in the same shape as the
// parallel path, instead of aborting lowering. One structured container
// exercises the shared classification for every effectful actor kind.

// FENCE: error: loom-lower-graph-memory: canonical Dataflow actor 'dataflow.fence' has no graph-region lowering
// FENCE-LABEL: dataflow.graph private @serial_fence
// FENCE: scf.if
// FENCE: dataflow.fence
// FENCE-NOT: dataflow.demux

//--- fence.mlir
dataflow.graph private @serial_fence(%start: none, %cond: i1) -> ()
    attributes {input_segments = array<i32: 1, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  scf.if %cond {
    %done = dataflow.fence %start
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
  }
  dataflow.graph.return %start : none
}

// ATOMIC: error: loom-lower-graph-memory: canonical Dataflow actor 'dataflow.atomic_rmw' has no graph-region lowering
// ATOMIC-LABEL: dataflow.graph private @serial_atomic
// ATOMIC: scf.if
// ATOMIC: dataflow.atomic_rmw
// ATOMIC-NOT: dataflow.demux

//--- atomic.mlir
dataflow.graph private @serial_atomic(
    %start: none, %cond: i1, %a: memref<10xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %v = arith.constant 7 : i32
  scf.if %cond {
    %old, %done = dataflow.atomic_rmw %a[%c0] %v %start
        {contract = #dataflow.rmw_contract<kind = add,
            access = <ordering = monotonic, sync_scope = <system>,
                      source_alignment_bytes = 4>>}
        : memref<10xi32>
  }
  dataflow.graph.return %start : none
}
