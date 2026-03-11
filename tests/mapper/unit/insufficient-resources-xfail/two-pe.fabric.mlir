// Fabric with only 2 addi PEs - a 3-operation DFG should fail to map.
// 3x2 mesh (6 switches, 2 PEs).

module {
  fabric.pe @pe_addi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
    %0 = arith.addi %a, %b : i32
    fabric.yield %0 : i32
  }

  fabric.module @two_pe(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>,
      %in3: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // Row 0
    %sw00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in0, %fifo_01_00, %fifo_10_00
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    %sw01:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in1, %sw00#0, %fifo_11_01
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    %sw02:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in2, %sw01#1, %fifo_12_02
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // Row 1
    %sw10:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in3, %sw00#1, %fifo_11_10
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    %sw11:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw01#2, %sw10#1, %fifo_12_11, %pe_cell00, %pe_cell01
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    %sw12:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw02#1, %sw11#2, %fifo_22_12_unused
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // FIFOs
    %fifo_01_00 = fabric.fifo [depth = 2] %sw01#0 : !dataflow.bits<32>
    %fifo_10_00 = fabric.fifo [depth = 2] %sw10#0 : !dataflow.bits<32>
    %fifo_11_01 = fabric.fifo [depth = 2] %sw11#0 : !dataflow.bits<32>
    %fifo_11_10 = fabric.fifo [depth = 2] %sw11#1 : !dataflow.bits<32>
    %fifo_12_02 = fabric.fifo [depth = 2] %sw12#0 : !dataflow.bits<32>
    %fifo_12_11 = fabric.fifo [depth = 2] %sw12#1 : !dataflow.bits<32>
    %fifo_22_12_unused = fabric.fifo [depth = 2] %sw12#2 : !dataflow.bits<32>

    // Only 2 PEs (insufficient for a 3-op DFG)
    %pe_cell00 = fabric.instance @pe_addi(%sw00#2, %sw10#3)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    %pe_cell01 = fabric.instance @pe_addi(%sw10#2, %sw11#3)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    fabric.yield %sw11#4 : !dataflow.bits<32>
  }
}
