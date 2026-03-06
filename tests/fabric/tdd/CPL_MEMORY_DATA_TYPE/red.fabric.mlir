// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_DATA_TYPE

// Store data port uses i64 but memref element type is i32.
fabric.module @bad_data(%staddr: !dataflow.bits<57>, %stdata: !dataflow.bits<64>) -> (none) {
  %stdone = fabric.memory
      [ldCount = 0, stCount = 1, lsqDepth = 1]
      (%staddr, %stdata)
      : memref<64xi32>, (!dataflow.bits<57>, !dataflow.bits<64>) -> (none)
  fabric.yield %stdone : none
}
