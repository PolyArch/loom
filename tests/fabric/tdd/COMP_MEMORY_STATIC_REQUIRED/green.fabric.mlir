// RUN: loom --adg %s

// A valid fabric.memory with a static memref shape.
fabric.module @static_mem(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<256xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
