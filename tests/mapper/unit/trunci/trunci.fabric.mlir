// Dual-mesh fabric: 64-bit input mesh + 32-bit output mesh, bridged by trunci PEs.

module {
  fabric.pe @pe_trunci(%arg0: !dataflow.bits<64>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i64):
    %0 = arith.trunci %a : i64 to i32
    fabric.yield %0 : i32
  }

  fabric.module @trunci(
      %in0: !dataflow.bits<64>
  ) -> (!dataflow.bits<32>) {

    // 64-bit input switches
    %sw0:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %in0, %fifo_1_0
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    %sw1:2 = fabric.switch [connectivity_table = [
        1, 1]]
        %sw0#0
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    // 64-bit backward FIFO
    %fifo_1_0 = fabric.fifo [depth = 2] %sw1#0 : !dataflow.bits<64>

    // trunci PEs (64-bit in, 32-bit out)
    %pe0 = fabric.instance @pe_trunci(%sw0#1)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<64>) -> !dataflow.bits<32>

    %pe1 = fabric.instance @pe_trunci(%sw1#1)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<64>) -> !dataflow.bits<32>

    // 32-bit output switches
    %sw_a:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe0, %fifo_b_a
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    %sw_b:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe1, %sw_a#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // 32-bit backward FIFO
    %fifo_b_a = fabric.fifo [depth = 2] %sw_b#0 : !dataflow.bits<32>

    fabric.yield %sw_b#1 : !dataflow.bits<32>
  }
}
