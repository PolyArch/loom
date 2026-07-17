// RUN: not loom-raise-opt --loom-lower-scf-to-dfg %s 2>&1 | FileCheck %s

llvm.mlir.global private @lookup_table(dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>) : !llvm.array<2 x f32>

func.func @global_lookup(%idx: i32) -> f32 {
  %base = llvm.mlir.addressof @lookup_table : !llvm.ptr
  %ptr = llvm.getelementptr %base[%idx] : (!llvm.ptr, i32) -> !llvm.ptr, f32
  %value = llvm.load %ptr : !llvm.ptr -> f32
  %scale = arith.constant 2.000000e+00 : f32
  %scaled = arith.mulf %value, %scale : f32
  return %scaled : f32
}

// CHECK: loom-lower-graph-memory: residual memory operation 'llvm.load' has no explicit completion event
