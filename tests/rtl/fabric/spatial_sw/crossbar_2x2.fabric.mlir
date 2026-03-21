// Test: fabric.spatial_sw 2x2 crossbar
fabric.module @test_spatial_sw_2x2(
  %in0: !fabric.bits<32>,
  %in1: !fabric.bits<32>
) -> (
  !fabric.bits<32>,
  !fabric.bits<32>
) {
  %out0, %out1 = fabric.spatial_sw %in0, %in1
    [connectivity_table = ["11", "11"]]
    {route_table = ["10", "01"]}
    : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield %out0, %out1 : !fabric.bits<32>, !fabric.bits<32>
}
