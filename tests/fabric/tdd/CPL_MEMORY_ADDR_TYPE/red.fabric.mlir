// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_ADDR_TYPE

// Load address port uses i32 instead of index.
fabric.module @bad_addr(%ldaddr: i32) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (i32) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
