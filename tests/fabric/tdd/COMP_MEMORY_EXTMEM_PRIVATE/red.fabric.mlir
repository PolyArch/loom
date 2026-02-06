// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: COMP_MEMORY_EXTMEM_PRIVATE

// fabric.extmemory with is_private attribute injected via generic syntax.
// The custom parser blocks is_private, so we use MLIR generic form inside a module.
fabric.module @bad_ext_private(%m: memref<?xi32>, %addr: index) -> (i32, none) {
  %lddata, %lddone = "fabric.extmemory"(%m, %addr) {
      memref_type = memref<?xi32>,
      ldCount = 1 : i64,
      stCount = 0 : i64,
      lsqDepth = 0 : i64,
      is_private = true
  } : (memref<?xi32>, index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
