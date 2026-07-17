// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: not loom-raise-opt --loom-lower-graph-memory %t/overlap.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=OVERLAP
// RUN: loom-raise-opt --loom-lower-graph-memory %t/non-i8.mlir -o %t/non-i8-lowered.mlir
// RUN: not loom-dfg-sim %t/non-i8-lowered.mlir --graph pointer_memcpy_non_i8 --memref 0=1.000000e+00,2.000000e+00 --memref 1=0,0 --output %t/non-i8.json 2>&1 | FileCheck %s --check-prefix=TYPE
// RUN: not loom-raise-opt --loom-lower-graph-memory %t/volatile.mlir -o /dev/null 2>&1 | FileCheck %s --check-prefix=VOLATILE

// OVERLAP: llvm.intr.memcpy overlapping ranges are unsupported
// TYPE: memory fixture type mismatch: existing i8, requested f32
// VOLATILE: volatile llvm.intr.memcpy is unsupported

//--- overlap.mlir
module {
  dataflow.graph.func private @pointer_memcpy_overlap(
      %ctrl: none, %base: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 0, 0, 1>,
                  result_segments = array<i32: 0, 0, 0>} {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %dst = llvm.getelementptr inbounds|nuw %base[1]
      : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%dst, %base, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }
}

//--- non-i8.mlir
module {
  dataflow.graph.func private @pointer_memcpy_non_i8(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %loaded = llvm.load %src {alignment = 4 : i64} : !llvm.ptr -> f32
    "llvm.intr.memcpy"(%dst, %src, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }
}

//--- volatile.mlir
module {
  dataflow.graph.func private @pointer_memcpy_volatile(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr) -> none
      attributes {input_segments = array<i32: 0, 0, 2>,
                  result_segments = array<i32: 0, 0, 0>} {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    "llvm.intr.memcpy"(%dst, %src, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }
}
