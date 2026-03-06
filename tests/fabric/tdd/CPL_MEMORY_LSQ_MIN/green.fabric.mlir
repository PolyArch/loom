// RUN: loom --adg %s

// A valid fabric.memory with stCount > 0 and lsqDepth >= 1.
fabric.module @valid_st(%ldaddr: !dataflow.bits<57>, %staddr: !dataflow.bits<57>, %stdata: !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none) {
  %lddata, %lddone, %stdone = fabric.memory
      [ldCount = 1, stCount = 1, lsqDepth = 4]
      (%ldaddr, %staddr, %stdata)
      : memref<64xi32>, (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none)
  fabric.yield %lddata, %lddone, %stdone : !dataflow.bits<32>, none, none
}
