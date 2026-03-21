// Test: fabric.map_tag with a 2-entry lookup table
fabric.module @test_map_tag(
  %in0: !fabric.tagged<!fabric.bits<32>, i2>
) -> (
  !fabric.tagged<!fabric.bits<32>, i3>
) {
  %retagged = fabric.map_tag %in0
    [table_size = 2 : i64]
    attributes {table = [[1 : i1, 0 : i2, 3 : i3],
                         [1 : i1, 1 : i2, 2 : i3]]}
    : !fabric.tagged<!fabric.bits<32>, i2>
      -> !fabric.tagged<!fabric.bits<32>, i3>
  fabric.yield %retagged : !fabric.tagged<!fabric.bits<32>, i3>
}
