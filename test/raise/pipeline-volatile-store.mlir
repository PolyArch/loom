// RUN: loom-raise-opt --loom-llvm-cf-to-cf --loom-lift-cf-to-scf --loom-llvm-arith-to-arith --canonicalize --loom-scf-while-to-for --canonicalize --loom-scf-for-to-forall --canonicalize %s | FileCheck %s

// A volatile llvm.store inside a counted loop body must SURVIVE the
// pipeline as a volatile store. The for-to-forall lift refuses to
// fire on volatile stores, so the loop stays as scf.for / scf.while.

// CHECK-LABEL: llvm.func @volatile_survives
// CHECK-NOT: scf.forall
// CHECK: llvm.store volatile
llvm.func @volatile_survives(%base: !llvm.ptr) {
    %c0 = llvm.mlir.constant(0 : i64) : i64
    %c1 = llvm.mlir.constant(1 : i64) : i64
    %c64 = llvm.mlir.constant(64 : i64) : i64
    %fzero = llvm.mlir.constant(0.0 : f32) : f32
    llvm.br ^bb_head(%c0 : i64)
  ^bb_head(%i: i64):
    %cond = llvm.icmp "slt" %i, %c64 : i64
    llvm.cond_br %cond, ^bb_body, ^bb_exit
  ^bb_body:
    %p = llvm.getelementptr inbounds %base[%i]
      : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    llvm.store volatile %fzero, %p : f32, !llvm.ptr
    %i_n = llvm.add %i, %c1 : i64
    llvm.br ^bb_head(%i_n : i64)
  ^bb_exit:
    llvm.return
}
