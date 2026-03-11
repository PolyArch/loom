// RUN: loom --adg %s

// A valid fabric.memory where data type (i32) matches memref element type.
fabric.module @valid_data(%stdata: !dataflow.bits<32>, %staddr: !dataflow.bits<57>, %ldaddr: !dataflow.bits<57>) -> (!dataflow.bits<32>, none, none) {
  %lddata, %lddone, %stdone = fabric.memory
      [ldCount = 1, stCount = 1, lsqDepth = 1]
      (%stdata, %staddr, %ldaddr)
      : memref<64xi32>, (!dataflow.bits<32>, !dataflow.bits<57>, !dataflow.bits<57>) -> (!dataflow.bits<32>, none, none)
  fabric.yield %lddata, %lddone, %stdone : !dataflow.bits<32>, none, none
}
