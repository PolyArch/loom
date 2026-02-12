// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_PRIVATE_OUTPUT

// A fabric.module yields a memref from a private fabric.memory (is_private = true).
fabric.module @private_export(%in_addr: index) -> (memref<64xi32>, i32, none) {
  %mem, %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0, is_private = true]
      (%in_addr)
      : memref<64xi32>, (index) -> (memref<64xi32>, i32, none)
  fabric.yield %mem, %lddata, %lddone : memref<64xi32>, i32, none
}
