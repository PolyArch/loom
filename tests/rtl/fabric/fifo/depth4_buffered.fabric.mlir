// Test: fabric.fifo with depth=4, buffered mode (not bypassable)
module {
  fabric.module @test_fifo_depth4(
    %in0: !fabric.bits<32>
  ) -> (
    !fabric.bits<32>
  ) {
    %out = fabric.fifo @fifo_0 [depth = 4] (%in0)
      : (!fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out : !fabric.bits<32>
  }
}
