// Test: fabric.spatial_sw 2x2 crossbar
module {
  fabric.module @test_spatial_sw_2x2(
    %in0: !fabric.bits<32>,
    %in1: !fabric.bits<32>
  ) -> (
    !fabric.bits<32>,
    !fabric.bits<32>
  ) {
    %sw:2 = fabric.spatial_sw @sw0 [connectivity_table = ["11", "11"]] (%in0, %in1)
      attributes {route_table = ["10", "01"]}
      : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %sw#0, %sw#1 : !fabric.bits<32>, !fabric.bits<32>
  }
}
