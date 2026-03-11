// Dual-mesh fabric: 32-bit input mesh + extsi PEs + 64-bit mesh with addi64 PE.

module {
  fabric.pe @pe_extsi(%arg0: !dataflow.bits<32>) -> (!dataflow.bits<64>) {
  ^bb0(%a: i32):
    %0 = arith.extsi %a : i32 to i64
    fabric.yield %0 : i64
  }
  fabric.pe @pe_addi64(%arg0: !dataflow.bits<64>, %arg1: !dataflow.bits<64>) -> (!dataflow.bits<64>) {
  ^bb0(%a: i64, %b: i64):
    %0 = arith.addi %a, %b : i64
    fabric.yield %0 : i64
  }

  fabric.module @mixed_width(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (!dataflow.bits<64>) {

    // 32-bit input switches
    %sw_in0:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %in0, %fifo_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    %sw_in1:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %in1, %sw_in0#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    %fifo_in = fabric.fifo [depth = 2] %sw_in1#0 : !dataflow.bits<32>

    // extsi PEs (32-bit in, 64-bit out)
    %pe_ext0 = fabric.instance @pe_extsi(%sw_in0#1)
        {sym_name = "pe_ext_0"}
        : (!dataflow.bits<32>) -> !dataflow.bits<64>

    %pe_ext1 = fabric.instance @pe_extsi(%sw_in1#1)
        {sym_name = "pe_ext_1"}
        : (!dataflow.bits<32>) -> !dataflow.bits<64>

    // 64-bit routing switches
    %sw_64a:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe_ext0, %fifo_64
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    %sw_64b:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %pe_ext1, %sw_64a#0
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %fifo_64 = fabric.fifo [depth = 2] %sw_64b#0 : !dataflow.bits<64>

    // addi64 PE
    %pe_add = fabric.instance @pe_addi64(%sw_64a#1, %sw_64b#1)
        {sym_name = "pe_add_0"}
        : (!dataflow.bits<64>, !dataflow.bits<64>) -> !dataflow.bits<64>

    // output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1, 1]]
        %pe_add, %sw_64b#2
        : !dataflow.bits<64>
       -> !dataflow.bits<64>

    fabric.yield %sw_out#0 : !dataflow.bits<64>
  }
}
