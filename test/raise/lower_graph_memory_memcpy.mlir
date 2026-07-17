// RUN: loom-raise-opt --loom-lower-graph-memory %s -o %t.lowered.mlir
// RUN: FileCheck %s --check-prefix=STRUCTURED-LOWERED < %t.lowered.mlir

// STRUCTURED-LOWERED-LABEL: dataflow.graph private @pointer_memcpy_structured_if
// STRUCTURED-LOWERED-NOT: llvm.intr.memcpy
// STRUCTURED-LOWERED-NOT: scf.if
// STRUCTURED-LOWERED-NOT: scf.for
// STRUCTURED-LOWERED: dataflow.demux
// STRUCTURED-LOWERED: dataflow.carry
// STRUCTURED-LOWERED: arith.cmpi ult
// STRUCTURED-LOWERED: dataflow.gate
// STRUCTURED-LOWERED: dataflow.load
// STRUCTURED-LOWERED: dataflow.store
// STRUCTURED-LOWERED-NOT: llvm.intr.memcpy
// STRUCTURED-LOWERED: dataflow.graph.return

module {
  dataflow.graph private @pointer_memcpy_structured_if(
      %ctrl: none, %copy_bytes: i32, %do_copy: i1, %src_offset: i32,
      %dst_offset: i32, %src: !llvm.ptr, %dst: !llvm.ptr) -> ()
      attributes {input_segments = array<i32: 4, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    scf.if %do_copy {
      %src_at = llvm.getelementptr %src[%src_offset]
          : (!llvm.ptr, i32) -> !llvm.ptr, i8
      %dst_at = llvm.getelementptr %dst[%dst_offset]
          : (!llvm.ptr, i32) -> !llvm.ptr, i8
      "llvm.intr.memcpy"(%dst_at, %src_at, %copy_bytes)
        <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
           isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    }
    dataflow.graph.return %ctrl : none
  }
}
