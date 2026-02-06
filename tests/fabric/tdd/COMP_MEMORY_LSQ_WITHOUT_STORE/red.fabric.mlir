// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MEMORY_LSQ_WITHOUT_STORE

// lsqDepth is nonzero but stCount is 0.
fabric.module @bad_lsq(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, lsqDepth = 4]
      (%ldaddr)
      : memref<64xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
