// RUN: loom-dfg-sim %s --graph pointer_memcpy_overlap --arg 0=none --memref 1=1,2,3 --output %t.overlap.json
// RUN: FileCheck %s --check-prefix=OVERLAP < %t.overlap.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_non_i8 --arg 0=none --memref 1=1.000000e+00,2.000000e+00 --memref 2=0,0 --output %t.type.json
// RUN: FileCheck %s --check-prefix=TYPE < %t.type.json
// RUN: loom-dfg-sim %s --graph pointer_memcpy_volatile --arg 0=none --memref 1=1,2 --memref 2=0,0 --output %t.volatile.json
// RUN: FileCheck %s --check-prefix=VOLATILE < %t.volatile.json

// OVERLAP-DAG: "status": "blocked"
// OVERLAP-DAG: "llvm.intr.memcpy overlapping ranges are unsupported"
// OVERLAP-NOT: "llvm.intr.memcpy": 1

// TYPE-DAG: "status": "blocked"
// TYPE-DAG: "memory fixture type mismatch: existing f32, requested i8"
// TYPE-NOT: "llvm.intr.memcpy": 1

// VOLATILE-DAG: "status": "blocked"
// VOLATILE-DAG: "volatile llvm.intr.memcpy is unsupported"
// VOLATILE-NOT: "llvm.intr.memcpy": 1

module {
  dataflow.graph.func private @pointer_memcpy_overlap(
      %ctrl: none, %base: !llvm.ptr) -> none {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %dst = llvm.getelementptr inbounds|nuw %base[1]
      : (!llvm.ptr) -> !llvm.ptr, i8
    "llvm.intr.memcpy"(%dst, %base, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_non_i8(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr) -> none {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    %loaded = llvm.load %src {alignment = 4 : i64} : !llvm.ptr -> f32
    "llvm.intr.memcpy"(%dst, %src, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }

  dataflow.graph.func private @pointer_memcpy_volatile(
      %ctrl: none, %src: !llvm.ptr, %dst: !llvm.ptr) -> none {
    %len = dataflow.constant %ctrl {const_value = 2 : i32} : i32
    "llvm.intr.memcpy"(%dst, %src, %len)
      <{arg_attrs = [{llvm.align = 1 : i64}, {llvm.align = 1 : i64}, {}],
         isVolatile = true}> : (!llvm.ptr, !llvm.ptr, i32) -> ()
    dataflow.graph.return %ctrl : none
  }
}
