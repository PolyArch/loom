// Test: fabric.temporal_sw with one slot routing tag=0 from in0 to out0
module {
  fabric.module @test_temporal_sw_1slot(
    %in0: !fabric.tagged<!fabric.bits<32>, i4>
  ) -> (
    !fabric.tagged<!fabric.bits<32>, i4>
  ) {
    %sw:1 = fabric.temporal_sw @tsw0 [num_route_table = 1] (%in0)
      attributes {
        route_table = [{tag = 0 : i64, input = 0 : i64, output = 0 : i64}]
      }
      : (!fabric.tagged<!fabric.bits<32>, i4>)
        -> (!fabric.tagged<!fabric.bits<32>, i4>)
    fabric.yield %sw#0 : !fabric.tagged<!fabric.bits<32>, i4>
  }
}
