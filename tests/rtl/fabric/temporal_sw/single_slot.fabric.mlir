// Test: fabric.temporal_sw with one slot routing tag=0 from in0 to out0
fabric.module @test_temporal_sw_1slot(
  %in0: !fabric.tagged<!fabric.bits<32>, i4>
) -> (
  !fabric.tagged<!fabric.bits<32>, i4>
) {
  %out = fabric.temporal_sw %in0
    [num_route_table = 1 : i64,
     connectivity_table = ["1"]]
    {route_table = ["1"]}
    : (!fabric.tagged<!fabric.bits<32>, i4>) -> (!fabric.tagged<!fabric.bits<32>, i4>)
  fabric.yield %out : !fabric.tagged<!fabric.bits<32>, i4>
}
