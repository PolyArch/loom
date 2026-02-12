// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_PORTS_EMPTY

// Both ldCount and stCount are 0.
fabric.module @no_ports() -> () {
  fabric.memory [ldCount = 0, stCount = 0]
      () : memref<64xi32>, () -> ()
  fabric.yield
}
