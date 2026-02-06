// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MEMORY_TAG_REQUIRED

// ldCount = 2 but load address ports use native index instead of tagged.
// Output layout: [lddata * 2] [lddone] = 3 outputs.
fabric.module @missing_tag(%ld0: index, %ld1: index) -> (i32, i32, none) {
  %d0, %d1, %done = fabric.memory
      [ldCount = 2, stCount = 0]
      (%ld0, %ld1)
      : memref<64xi32>, (index, index) -> (i32, i32, none)
  fabric.yield %d0, %d1, %done : i32, i32, none
}
