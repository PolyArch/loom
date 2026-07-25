// RUN: split-file %s %t
// RUN: loom-raise-opt --loom-llvm-arith-to-arith %t/plain.mlir | FileCheck %s --check-prefix=PLAIN
// RUN: loom-raise-opt --loom-llvm-arith-to-arith %t/plain.mlir | mlir-opt --convert-arith-to-llvm | mlir-translate --mlir-to-llvmir -o %t/plain.ll
// RUN: loom-raise-opt --loom-llvm-arith-to-arith %t/flagged.mlir | FileCheck %s --check-prefix=FLAGGED --implicit-check-not=arith.extf --implicit-check-not=arith.truncf
// RUN: loom-raise-opt --loom-llvm-arith-to-arith %t/flagged.mlir | mlir-opt --convert-arith-to-llvm | mlir-translate --mlir-to-llvmir -o %t/flagged.ll

// A float resize is the one alias that must be checked through the production
// lowering, not only in the raising direction: the pinned arith-to-llvm
// lowering of a fast-math arith.extf or arith.truncf does not carry the
// contract back onto the llvm op, so the raised form leaves a foreign arith
// fast-math attribute that mlir-translate rejects. An unflagged resize raises
// and lowers cleanly; a flagged resize is preserved in llvm form and still
// lowers cleanly.

// An unflagged llvm.fpext / llvm.fptrunc restates exactly what arith.extf /
// arith.truncf lower back to, so it is raised and the whole module lowers to
// LLVM IR that mlir-translate accepts.
// PLAIN-LABEL: llvm.func @plain
// PLAIN: arith.extf
// PLAIN: arith.truncf

// Flagged resizes keep their llvm spellings and still lower to LLVM IR.
// FLAGGED-LABEL: llvm.func @flagged
// FLAGGED: llvm.fpext {{.*}} fastmath<nnan> : f32 to f64
// FLAGGED: llvm.fptrunc {{.*}} fastmath<ninf> : f64 to f32

//--- plain.mlir
llvm.func @plain(%s: f32, %d: f64) -> f64 {
  %e = llvm.fpext %s : f32 to f64
  %t = llvm.fptrunc %d : f64 to f32
  llvm.return %e : f64
}

//--- flagged.mlir
llvm.func @flagged(%s: f32, %d: f64) -> f64 {
  %e = llvm.fpext %s fastmath<nnan> : f32 to f64
  %t = llvm.fptrunc %d fastmath<ninf> : f64 to f32
  llvm.return %e : f64
}
