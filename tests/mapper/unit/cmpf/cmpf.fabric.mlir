// Dual-mesh fabric: 32-bit input mesh + 1-bit output mesh, bridged by cmpf PEs.

module {
  fabric.pe @pe_cmpf(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<1>) {
  ^bb0(%a: f32, %b: f32):
    %0 = arith.cmpf olt, %a, %b : f32
    fabric.yield %0 : i1
  }

  fabric.module @cmpf(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (!dataflow.bits<1>, !dataflow.bits<1>) {

    // 32-bit input switches
    %sw0:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in0, %fifo_1_0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    %sw1:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in1, %sw0#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // 32-bit backward FIFO
    %fifo_1_0 = fabric.fifo [depth = 2] %sw1#0 : !dataflow.bits<32>

    // cmpf PEs (32,32 -> 1)
    %pe0 = fabric.instance @pe_cmpf(%sw0#1, %sw1#1)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>

    %pe1 = fabric.instance @pe_cmpf(%sw0#2, %sw1#2)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>

    // 1-bit output switches
    %sw_a:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe0, %fifo_b_a
        : !dataflow.bits<1>
       -> !dataflow.bits<1>, !dataflow.bits<1>

    %sw_b:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe1, %sw_a#0
        : !dataflow.bits<1>
       -> !dataflow.bits<1>, !dataflow.bits<1>

    // 1-bit backward FIFO
    %fifo_b_a = fabric.fifo [depth = 2] %sw_b#0 : !dataflow.bits<1>

    fabric.yield %sw_a#1, %sw_b#1 : !dataflow.bits<1>, !dataflow.bits<1>
  }
}
