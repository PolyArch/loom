// Test: fabric.add_tag with constant tag value
fabric.module @test_add_tag(
  %in0: !fabric.bits<32>
) -> (
  !fabric.tagged<!fabric.bits<32>, i4>
) {
  %tagged = fabric.add_tag %in0 {tag = 5 : i4}
    : !fabric.bits<32> -> !fabric.tagged<!fabric.bits<32>, i4>
  fabric.yield %tagged : !fabric.tagged<!fabric.bits<32>, i4>
}
