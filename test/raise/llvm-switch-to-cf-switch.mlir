// RUN: loom-raise-opt --loom-llvm-cf-to-cf %s | FileCheck %s

// llvm.switch is rewritten into cf.switch so the region-level CFG-to-SCF
// transformation can recognize its branch structure. Default destination,
// case values, and case operands carry through directly.

// CHECK-LABEL: llvm.func @route_value
// CHECK-NOT: llvm.switch
// CHECK: cf.switch
llvm.func @route_value(%v: i32) -> i32 {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c2 = llvm.mlir.constant(2 : i32) : i32
    llvm.switch %v : i32, ^bb_default [
      0: ^bb0,
      1: ^bb1
    ]
  ^bb_default:
    llvm.return %c0 : i32
  ^bb0:
    llvm.return %c1 : i32
  ^bb1:
    llvm.return %c2 : i32
}
