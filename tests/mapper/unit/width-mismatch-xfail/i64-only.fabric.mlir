// Fabric with only i64-width PEs and switches - i32 DFG should fail to map.

module {
  fabric.pe @pe_addi64(%arg0: !dataflow.bits<64>, %arg1: !dataflow.bits<64>) -> (!dataflow.bits<64>) {
  ^bb0(%a: i64, %b: i64):
    %0 = arith.addi %a, %b : i64
    fabric.yield %0 : i64
  }

  fabric.module @i64_only(
      %in0: !dataflow.bits<64>,
      %in1: !dataflow.bits<64>,
      %in2: !dataflow.bits<64>
  ) -> (!dataflow.bits<64>, !dataflow.bits<64>) {

    %sw00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in0, %fifo_01_00, %fifo_10_00
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %sw01:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw00#0, %fifo_02_01, %fifo_11_01
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %sw02:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in1, %sw01#1, %fifo_12_02
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    %sw10:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw00#1, %fifo_11_10, %fifo_20_10
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>,
          !dataflow.bits<64>, !dataflow.bits<64>

    %sw11:6 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw01#2, %sw10#1, %fifo_12_11, %fifo_21_11, %pe_cell00
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>,
          !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %sw12:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw02#1, %sw11#2, %fifo_22_12, %pe_cell01
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %sw20:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in2, %sw10#2, %fifo_21_20
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %sw21:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw11#3, %sw20#1, %fifo_22_21, %pe_cell10
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %sw22:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw12#2, %sw21#2, %pe_cell11
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    %fifo_01_00 = fabric.fifo [depth = 2] %sw01#0 : !dataflow.bits<64>
    %fifo_02_01 = fabric.fifo [depth = 2] %sw02#0 : !dataflow.bits<64>
    %fifo_11_10 = fabric.fifo [depth = 2] %sw11#1 : !dataflow.bits<64>
    %fifo_12_11 = fabric.fifo [depth = 2] %sw12#1 : !dataflow.bits<64>
    %fifo_21_20 = fabric.fifo [depth = 2] %sw21#1 : !dataflow.bits<64>
    %fifo_22_21 = fabric.fifo [depth = 2] %sw22#1 : !dataflow.bits<64>
    %fifo_10_00 = fabric.fifo [depth = 2] %sw10#0 : !dataflow.bits<64>
    %fifo_11_01 = fabric.fifo [depth = 2] %sw11#0 : !dataflow.bits<64>
    %fifo_12_02 = fabric.fifo [depth = 2] %sw12#0 : !dataflow.bits<64>
    %fifo_20_10 = fabric.fifo [depth = 2] %sw20#0 : !dataflow.bits<64>
    %fifo_21_11 = fabric.fifo [depth = 2] %sw21#0 : !dataflow.bits<64>
    %fifo_22_12 = fabric.fifo [depth = 2] %sw22#0 : !dataflow.bits<64>

    %pe_cell00 = fabric.instance @pe_addi64(%sw00#2, %sw10#3)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<64>, !dataflow.bits<64>) -> !dataflow.bits<64>
    %pe_cell01 = fabric.instance @pe_addi64(%sw01#3, %sw11#4)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<64>, !dataflow.bits<64>) -> !dataflow.bits<64>
    %pe_cell10 = fabric.instance @pe_addi64(%sw10#4, %sw20#2)
        {sym_name = "pe_1_0"}
        : (!dataflow.bits<64>, !dataflow.bits<64>) -> !dataflow.bits<64>
    %pe_cell11 = fabric.instance @pe_addi64(%sw11#5, %sw21#3)
        {sym_name = "pe_1_1"}
        : (!dataflow.bits<64>, !dataflow.bits<64>) -> !dataflow.bits<64>

    fabric.yield %sw22#2, %sw12#3 : !dataflow.bits<64>, !dataflow.bits<64>
  }
}
