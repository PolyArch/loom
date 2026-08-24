// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading %t.dir/fence.mlir | FileCheck %s --check-prefix=FENCE
// RUN: loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading %t.dir/atomic.mlir | FileCheck %s --check-prefix=ATOMIC
// RUN: loom-raise-opt --loom-lower-graph-memory --mlir-disable-threading %t.dir/source.mlir | FileCheck %s --check-prefix=SOURCE

// Atomic and fence actors use the same graph-region completion lowering as
// ordinary memory actors. One structured container exercises the shared
// classification and memory frontier path for each effectful actor kind.

// FENCE-LABEL: dataflow.graph private @serial_fence
// FENCE-NOT: scf.if
// FENCE: dataflow.fence

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

// ATOMIC-LABEL: dataflow.graph private @serial_atomic
// ATOMIC-NOT: scf.if
// ATOMIC: dataflow.atomic_rmw

// SOURCE-LABEL: dataflow.graph private @source_atomic
// SOURCE: dataflow.atomic_rmw
// SOURCE: dataflow.cmpxchg
// SOURCE: dataflow.fence
// SOURCE: dataflow.plain_access

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

//--- source.mlir
module attributes {
  llvm.data_layout = "e-p:64:64",
  dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>
} {
  dataflow.graph private @source_atomic(
      %start: none, %base: !llvm.ptr, %expected: i32, %desired: i32,
      %index: i64) -> ()
      attributes {input_segments = array<i32: 4, 0, 0>,
                  result_segments = array<i32: 0, 0, 0>} {
    %ptr = llvm.getelementptr inbounds %base[%index]
        : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %old = llvm.atomicrmw add %ptr, %desired monotonic {alignment = 4 : i64}
        : !llvm.ptr, i32
    %pair = llvm.cmpxchg volatile %ptr, %expected, %desired
        syncscope("singlethread") acq_rel monotonic {alignment = 4 : i64}
        : !llvm.ptr, i32
    %old_pair = llvm.extractvalue %pair[0] : !llvm.struct<(i32, i1)>
    %success = llvm.extractvalue %pair[1] : !llvm.struct<(i32, i1)>
    llvm.fence seq_cst
    llvm.store volatile %old_pair, %ptr {alignment = 4 : i64}
        : i32, !llvm.ptr
    llvm.store %success, %ptr : i1, !llvm.ptr
    dataflow.graph.return %start : none
  }
}
