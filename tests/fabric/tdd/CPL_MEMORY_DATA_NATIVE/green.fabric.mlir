// RUN: loom --adg %s

// Memory data port uses bits<32> (correct).
fabric.module @good_mem_data(
    %ldaddr: !dataflow.bits<57>
) -> (!dataflow.bits<32>, none) {
  %lddata, %lddone = fabric.memory
      [ldCount = 1, stCount = 0]
      (%ldaddr)
      : memref<64xi32>, (!dataflow.bits<57>) -> (!dataflow.bits<32>, none)
  fabric.yield %lddata, %lddone : !dataflow.bits<32>, none
}
