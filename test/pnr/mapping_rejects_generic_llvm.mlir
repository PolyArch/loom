// RUN: rm -rf %t.dir
// RUN: split-file %s %t.dir
// RUN: not loom-pnr-map --dfg-mlir %t.dir/intrinsic.mlir \
// RUN:   --graph generic_intrinsic --hardware-mlir %t.dir/hardware.mlir \
// RUN:   --hardware empty --workload generic_intrinsic --output %t.csv \
// RUN:   2>&1 | FileCheck %s --check-prefix=INTRINSIC
// RUN: not loom-pnr-map --dfg-mlir %t.dir/asm.mlir \
// RUN:   --graph generic_inline_asm --hardware-mlir %t.dir/hardware.mlir \
// RUN:   --hardware empty --workload generic_inline_asm --output %t.asm.csv \
// RUN:   2>&1 | FileCheck %s --check-prefix=ASM

// INTRINSIC: finalized graph contains unregistered actor 'llvm.call_intrinsic'
// ASM: finalized graph contains unregistered actor 'llvm.inline_asm'

//--- intrinsic.mlir
module {
  dataflow.graph private @generic_intrinsic(
      %start: none, %lhs: i32, %rhs: i32) -> (i32)
      attributes {input_segments = array<i32: 2, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result = llvm.call_intrinsic "llvm.arm.qadd16"(%lhs, %rhs)
        : (i32, i32) -> i32
    dataflow.graph.return %start, %result : none, i32
  }
}

//--- asm.mlir
module {
  dataflow.graph private @generic_inline_asm(
      %start: none, %lhs: i32, %rhs: i32, %amount: i32) -> (i32)
      attributes {input_segments = array<i32: 3, 0, 0>,
                  result_segments = array<i32: 1, 0, 0>} {
    %result = llvm.inline_asm tail_call_kind = <tail> asm_dialect = att
        "pkhbt $0, $1, $2, lsl $3", "=r,r,r,I" %lhs, %rhs, %amount
        : (i32, i32, i32) -> i32
    dataflow.graph.return %start, %result : none, i32
  }
}

//--- hardware.mlir
fabric.module @empty() {
  fabric.yield
}
