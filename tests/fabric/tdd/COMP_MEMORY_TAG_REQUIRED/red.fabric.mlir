// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MEMORY_TAG_REQUIRED

// ldCount = 2 requires tagged ports, but native index is provided.
// Singular port layout: 1 input (ld_addr), 2 outputs (ld_data, ld_done).
fabric.module @missing_tag(%ld_addr: index) -> (i32, none) {
  %ld_data, %ld_done = fabric.memory
      [ldCount = 2, stCount = 0]
      (%ld_addr)
      : memref<64xi32>, (index) -> (i32, none)
  fabric.yield %ld_data, %ld_done : i32, none
}
