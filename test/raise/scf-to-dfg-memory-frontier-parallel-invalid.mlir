// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/forged.mlir 2>&1 | FileCheck %s --check-prefix=FORGED
// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/overlap.mlir 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/pointer-overlap.mlir 2>&1 | FileCheck %s --check-prefix=POINTER-OVERLAP
// RUN: not loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading --mlir-print-ir-after-failure --mlir-print-ir-module-scope %t.dir/unsupported-actor.mlir 2>&1 | FileCheck %s --check-prefix=ACTOR

// FORGED: error: loom-lower-graph-memory: parallel SCF carries unsupported author metadata
// FORGED-LABEL: dataflow.graph private @forged_parallel
// FORGED: scf.parallel
// FORGED: memref.store
// FORGED: loom.parallel_group
// FORGED: loom.parallel_schedule
// FORGED-NOT: dataflow.store

//--- forged.mlir
dataflow.graph private @forged_parallel(
    %start: none, %a: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %value = arith.constant 7 : i32
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    memref.store %value, %a[%i] : memref<?xi32>
    scf.reduce
  } {loom.parallel_group = 7 : i64, loom.parallel_schedule}
  dataflow.graph.return %start : none
}

//--- overlap.mlir

// OVERLAP: error: loom-lower-graph-memory: parallel lanes have overlapping plain memory effects
// OVERLAP-LABEL: dataflow.graph private @would_be_rewritten
// OVERLAP: memref.load
// OVERLAP-NOT: dataflow.load
// OVERLAP-LABEL: dataflow.graph private @overlapping_parallel
// OVERLAP: scf.parallel
// OVERLAP: memref.store
// OVERLAP-NOT: dataflow.store
dataflow.graph private @would_be_rewritten(
    %start: none, %index: index, %a: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

dataflow.graph private @overlapping_parallel(
    %start: none, %a: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 0, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %value = arith.constant 7 : i32
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    memref.store %value, %a[%c0] : memref<?xi32>
    scf.reduce
  }
  dataflow.graph.return %start : none
}

//--- pointer-overlap.mlir

// POINTER-OVERLAP: error: loom-lower-graph-memory: parallel lanes have overlapping plain memory byte ranges
// POINTER-OVERLAP-LABEL: dataflow.graph private @overlapping_pointer_parallel
// POINTER-OVERLAP: llvm.store
// POINTER-OVERLAP-NOT: dataflow.store
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>
} {
  dataflow.graph private @overlapping_pointer_parallel(
      %ctrl: none, %pointer: !llvm.ptr) -> ()
      attributes {input_segments = array<i32: 1, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %zero = arith.constant 0 : i64
    %value = arith.constant 7 : i32
    scf.forall (%lane) = (%c0) to (%c2) step (%c1) {
      %address = llvm.getelementptr inbounds %pointer[%zero]
          : (!llvm.ptr, i64) -> !llvm.ptr, i32
      llvm.store %value, %address : i32, !llvm.ptr
    }
    dataflow.graph.return %ctrl : none
  }
}

//--- unsupported-actor.mlir

// ACTOR: error: loom-lower-graph-memory: parallel actor 'dataflow.fence' has no completion lowering
// ACTOR-LABEL: dataflow.graph private @would_be_rewritten
// ACTOR: memref.load
// ACTOR-NOT: dataflow.load
// ACTOR-LABEL: dataflow.graph private @parallel_fence
// ACTOR: scf.parallel
// ACTOR: dataflow.fence
dataflow.graph private @would_be_rewritten(
    %start: none, %index: index, %a: memref<?xi32>) -> ()
    attributes {input_segments = array<i32: 1, 0, 1>,
                result_segments = array<i32: 0, 0, 0>} {
  %value = memref.load %a[%index] : memref<?xi32>
  dataflow.graph.return %start : none
}

dataflow.graph private @parallel_fence(%start: none) -> ()
    attributes {input_segments = array<i32: 0, 0, 0>,
                result_segments = array<i32: 0, 0, 0>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.parallel (%i) = (%c0) to (%c2) step (%c1) {
    %done = dataflow.fence %start
        {contract = #dataflow.fence_contract<ordering = seq_cst,
                                             sync_scope = <system>>}
    scf.reduce
  }
  dataflow.graph.return %start : none
}
