// Dual-mesh fabric: 1-bit condition mesh + 32-bit data mesh, with select PEs.

module {
  fabric.pe @pe_select(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>,
                       %arg2: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%cond: i1, %a: i32, %b: i32):
    %0 = arith.select %cond, %a, %b : i32
    fabric.yield %0 : i32
  }

  fabric.module @select(
      %cond_in: !dataflow.bits<1>,
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>, !dataflow.bits<32>) {

    // 1-bit switches for condition routing
    %sw_c0:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %cond_in, %fifo_c1_c0
        : !dataflow.bits<1>
       -> !dataflow.bits<1>, !dataflow.bits<1>

    %sw_c1:2 = fabric.switch [connectivity_table = [
        1, 1]]
        %sw_c0#0
        : !dataflow.bits<1>
       -> !dataflow.bits<1>, !dataflow.bits<1>

    %fifo_c1_c0 = fabric.fifo [depth = 2] %sw_c1#0 : !dataflow.bits<1>

    // 32-bit switches for data routing
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

    %fifo_1_0 = fabric.fifo [depth = 2] %sw1#0 : !dataflow.bits<32>

    // select PEs (1-bit cond + 32-bit data -> 32-bit output)
    %pe0 = fabric.instance @pe_select(%sw_c0#1, %sw0#1, %sw1#1)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    %pe1 = fabric.instance @pe_select(%sw_c1#1, %sw0#2, %sw1#2)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

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

    %fifo_b_a = fabric.fifo [depth = 2] %sw_b#0 : !dataflow.bits<32>

    fabric.yield %sw_a#1, %sw_b#1 : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
