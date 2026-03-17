// Minimal ADG with 2 spatial_sw, 1 spatial_pe, and 1 fabric.extmemory.
// The PE exposes only handshake.load so the sample exercises extmemory
// routing without requiring multiple PEs.
module {
  fabric.spatial_sw @test_sw [connectivity_table = ["11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111", "11111111"]] : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)

  fabric.spatial_pe @test_pe(%p0: !fabric.bits<64>, %p1: !fabric.bits<64>, %p2: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @fu_load(%arg0: index, %arg1: i32, %arg2: none) -> (i32, index) [latency = 1, interval = 1] {
      %0, %1 = handshake.load [%arg0] %arg1, %arg2 : index, i32
      fabric.yield %0, %1 : i32, index
    }
    fabric.yield
  }

  fabric.module @sw2_pe1_extmem1_test(%mem0: memref<?xi32>, %idx: !fabric.bits<64>, %ctrl: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    %ext0:2 = fabric.extmemory @extmem0 [ldCount = 1, stCount = 0, lsqDepth = 0, memrefType = memref<?xi32>] (%mem0, %sw0#4) : (memref<?xi32>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>)

    %sw0:8 = fabric.instance @test_sw(%sw1#4, %idx, %ctrl, %sw1#5, %ext0#0, %ext0#1, %sw0#6, %sw0#7) {sym_name = "sw_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)

    %sw1:8 = fabric.instance @test_sw(%pe#0, %pe#1, %sw0#0, %sw0#1, %sw0#2, %sw0#5, %sw1#6, %sw1#7) {sym_name = "sw_1"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)

    %pe:2 = fabric.instance @test_pe(%sw1#0, %sw1#1, %sw1#2) {sym_name = "pe_0"} : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>)

    fabric.yield %sw1#3, %sw0#3 : !fabric.bits<64>, !fabric.bits<64>
  }
}
