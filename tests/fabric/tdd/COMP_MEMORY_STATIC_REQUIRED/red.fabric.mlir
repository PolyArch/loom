// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MEMORY_STATIC_REQUIRED

// fabric.memory uses a dynamic memref shape (? dimension).
fabric.module @dynamic_mem(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<?xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
