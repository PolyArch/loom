// RUN: loom-raise-opt --loom-llvm-func-to-func --loom-llvm-cf-to-cf --lift-cf-to-scf --loom-llvm-arith-to-arith --canonicalize %s | FileCheck %s

// End-to-end: an llvm.switch in the input survives translation through
// loom-llvm-cf-to-cf into cf.switch and is then lifted by
// --lift-cf-to-scf into scf.index_switch.

// CHECK-LABEL: func.func @switch_to_index
// CHECK-NOT: llvm.switch
// CHECK-NOT: cf.switch
// CHECK: scf.index_switch
llvm.func @switch_to_index(%v: i32, %a: i32, %b: i32, %c: i32) -> i32 {
    llvm.switch %v : i32, ^bb_def [
      0: ^bb0,
      1: ^bb1
    ]
  ^bb_def:
    llvm.br ^bb_join(%a : i32)
  ^bb0:
    llvm.br ^bb_join(%b : i32)
  ^bb1:
    llvm.br ^bb_join(%c : i32)
  ^bb_join(%r: i32):
    llvm.return %r : i32
}
