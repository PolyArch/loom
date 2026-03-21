// Test: fabric.fifo with depth=4, buffered mode (not bypassable)
fabric.module @test_fifo_depth4(
  %in0: !fabric.bits<32>
) -> (
  !fabric.bits<32>
) {
  %out = fabric.fifo %in0 [depth = 4 : i64]
    : !fabric.bits<32> -> !fabric.bits<32>
  fabric.yield %out : !fabric.bits<32>
}
