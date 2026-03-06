// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_ADDR_TYPE

// Load address port uses i32 instead of index.
fabric.module @bad_addr(%ldaddr: !dataflow.bits<32>) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (!dataflow.bits<32>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
