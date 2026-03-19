// Minimal ADG: 1 spatial_sw + 1 spatial_pe
module {
  fabric.spatial_sw @test_sw [connectivity_table = ["11111", "11111", "11111", "11111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
  fabric.spatial_pe @test_pe(%p0: !fabric.bits<64>, %p1: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @fu_add(%arg0: i32, %arg1: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.yield
  }
  fabric.module @sw1_pe1_test(%in0: !fabric.bits<64>, %in1: !fabric.bits<64>, %in2: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    %sw:4 = fabric.instance @test_sw(%in0, %in1, %in2, %pe#0, %pe#1) {sym_name = "sw_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
    %pe:2 = fabric.instance @test_pe(%sw#0, %sw#1) {sym_name = "pe_0"} : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>)
    fabric.yield %sw#2, %sw#3 : !fabric.bits<64>, !fabric.bits<64>
  }
}
