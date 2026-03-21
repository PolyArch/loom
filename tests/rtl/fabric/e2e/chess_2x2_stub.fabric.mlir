// End-to-end stub: minimal 2x2 chess grid with 2 spatial PEs, 2 spatial
// switches, and 1 FIFO. Serves as a placeholder for full mapped test.
//
// Topology:
//   in0 -> sw0 -> pe0 (addi) -> fifo0 -> sw1 -> pe1 (muli) -> out0
//   in1 -> sw0 (2nd port)   ----^
//   in2 -> sw1 (2nd port)   -----> pe1 (2nd input)
module {
  // -- Function units --
  fabric.function_unit @fu_add(%a: i32, %b: i32) -> (i32)
      [latency = 1, interval = 1] {
    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }

  fabric.function_unit @fu_mul(%a: i32, %b: i32) -> (i32)
      [latency = 1, interval = 1] {
    %r = arith.muli %a, %b : i32
    fabric.yield %r : i32
  }

  // -- Spatial PEs --
  fabric.spatial_pe @pe_add(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.instance @fu_add() {sym_name = "fu_add_0"} : () -> ()
    fabric.yield
  }

  fabric.spatial_pe @pe_mul(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.instance @fu_mul() {sym_name = "fu_mul_0"} : () -> ()
    fabric.yield
  }

  // -- Top-level module: 2x2 chess grid --
  fabric.module @chess_2x2_stub(
      %in0: !fabric.bits<32>,
      %in1: !fabric.bits<32>,
      %in2: !fabric.bits<32>)
      -> (!fabric.bits<32>) {

    // Switch 0: 2-input 2-output crossbar
    // Routes in0 -> pe_add input0, in1 -> pe_add input1
    %sw0_out0, %sw0_out1 = fabric.spatial_sw %in0, %in1
      [connectivity_table = ["11", "11"]]
      {route_table = ["10", "01"]}
      : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)

    // PE 0: addi
    %pe0_out = fabric.instance @pe_add(%sw0_out0, %sw0_out1) {sym_name = "pe_0"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)

    // FIFO between PE0 and SW1
    %fifo_out = fabric.fifo %pe0_out#0 [depth = 4 : i64]
        : !fabric.bits<32> -> !fabric.bits<32>

    // Switch 1: 2-input 2-output crossbar
    // Routes fifo_out -> pe_mul input0, in2 -> pe_mul input1
    %sw1_out0, %sw1_out1 = fabric.spatial_sw %fifo_out, %in2
      [connectivity_table = ["11", "11"]]
      {route_table = ["10", "01"]}
      : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)

    // PE 1: muli
    %pe1_out = fabric.instance @pe_mul(%sw1_out0, %sw1_out1) {sym_name = "pe_1"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)

    fabric.yield %pe1_out#0 : !fabric.bits<32>
  }
}
