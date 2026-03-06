// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_MEMORY_LSQ_MIN

// stCount > 0 but lsqDepth is 0 (less than 1).
fabric.module @bad_lsq_min(%ldaddr: !dataflow.bits<57>, %staddr: !dataflow.bits<57>, %stdata: !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none) {
  %lddata, %lddone, %stdone = fabric.memory
      [ldCount = 1, stCount = 1, lsqDepth = 0]
      (%ldaddr, %staddr, %stdata)
      : memref<64xi32>, (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none)
  fabric.yield %lddata, %lddone, %stdone : !dataflow.bits<32>, none, none
}
