// Dual-mesh fabric: 32-bit input mesh + 64-bit output mesh, bridged by extsi PEs.

module {
  fabric.pe @pe_extsi(%arg0: !dataflow.bits<32>) -> (!dataflow.bits<64>) {
  ^bb0(%a: i32):
    %0 = arith.extsi %a : i32 to i64
    fabric.yield %0 : i64
  }

  fabric.module @extsi(
      %in0: !dataflow.bits<32>
  ) -> (!dataflow.bits<64>) {

    // 32-bit input switches
    %sw0:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %in0, %fifo_1_0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    %sw1:2 = fabric.switch [connectivity_table = [
        1, 1]]
        %sw0#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // 32-bit backward FIFO
    %fifo_1_0 = fabric.fifo [depth = 2] %sw1#0 : !dataflow.bits<32>

    // extsi PEs (32-bit in, 64-bit out)
    %pe0 = fabric.instance @pe_extsi(%sw0#1)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>) -> !dataflow.bits<64>

    %pe1 = fabric.instance @pe_extsi(%sw1#1)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<32>) -> !dataflow.bits<64>

    // 64-bit output switches
    %sw_a:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe0, %fifo_b_a
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    %sw_b:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe1, %sw_a#0
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    // 64-bit backward FIFO
    %fifo_b_a = fabric.fifo [depth = 2] %sw_b#0 : !dataflow.bits<64>

    fabric.yield %sw_b#1 : !dataflow.bits<64>
  }
}
