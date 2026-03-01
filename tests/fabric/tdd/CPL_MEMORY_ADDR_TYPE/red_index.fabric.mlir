// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_ADDR_TYPE

// Memory address port uses native index type (not bits<ADDR_BIT_WIDTH>).
fabric.module @bad_index_addr(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
