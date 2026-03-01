// RUN: loom --adg %s

// A valid tagged load PE: consistent tag widths (i4) across all ports.
fabric.module @test(
    %addr: !dataflow.tagged<!dataflow.bits<57>, i4>,
    %data: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %ctrl: !dataflow.tagged<none, i4>
) -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<57>, i4>) {
  %d, %a = fabric.pe %addr, %data, %ctrl
      [lqDepth = 4]
      : (!dataflow.tagged<!dataflow.bits<57>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>,
         !dataflow.tagged<none, i4>)
      -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<57>, i4>) {
  ^bb0(%x: index, %y: i32, %c: none):
    %ld_d, %ld_a = handshake.load [%x] %y, %c : index, i32
    fabric.yield %ld_d, %ld_a : i32, index
  }
  fabric.yield %d, %a : !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<57>, i4>
}
