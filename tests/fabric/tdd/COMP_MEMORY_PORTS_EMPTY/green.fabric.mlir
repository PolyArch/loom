// RUN: loom --adg %s

// A valid fabric.memory with at least one port (ldCount = 1), inline form.
fabric.module @valid_ld_only(%ldaddr: index) -> (i32, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (index) -> (i32, none)
  fabric.yield %lddata, %lddone : i32, none
}
