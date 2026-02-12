// RUN: loom --adg %s

// A fabric.module with inline fabric.extmemory where the memref operand
// is a block argument of the parent module.
fabric.module @valid_extmem(%m: memref<?xi32>, %addr: index) -> (i32, none) {
  %lddata, %lddone = fabric.extmemory
      [ldCount = 1, stCount = 0]
      (%m, %addr)
      : memref<?xi32>, (memref<?xi32>, index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
