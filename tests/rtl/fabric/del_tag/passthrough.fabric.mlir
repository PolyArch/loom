// Test: fabric.del_tag strips tag from tagged value
fabric.module @test_del_tag(
  %in0: !fabric.tagged<!fabric.bits<32>, i4>
) -> (
  !fabric.bits<32>
) {
  %untagged = fabric.del_tag %in0
    : !fabric.tagged<!fabric.bits<32>, i4> -> !fabric.bits<32>
  fabric.yield %untagged : !fabric.bits<32>
}
