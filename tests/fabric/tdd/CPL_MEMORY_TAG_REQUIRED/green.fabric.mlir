// RUN: loom --adg %s

// A valid fabric.memory with ldCount = 2, tagged address and data ports.
// Output layout: [lddata * 2] [lddone]
fabric.module @valid_tagged(
    %ld0: !dataflow.tagged<!dataflow.bits<57>, i1>,
    %ld1: !dataflow.tagged<!dataflow.bits<57>, i1>
) -> (!dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<none, i1>) {
  %d0, %d1, %done = fabric.memory
      [ldCount = 2, stCount = 0]
      (%ld0, %ld1)
      : memref<64xi32>,
        (!dataflow.tagged<!dataflow.bits<57>, i1>, !dataflow.tagged<!dataflow.bits<57>, i1>)
        -> (!dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<!dataflow.bits<32>, i1>,
            !dataflow.tagged<none, i1>)
  fabric.yield %d0, %d1, %done
      : !dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<!dataflow.bits<32>, i1>, !dataflow.tagged<none, i1>
}
