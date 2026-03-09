// ADG: 3x3 32-bit mesh + 2x2 64-bit mesh, bridged by extsi PEs.
//
// 32-bit mesh (9 switches, 4 cells):
//       col0        col1        col2
// row0: SW00 ------ SW01 ------ SW02
//        |  Cell00   |  Cell01   |
// row1: SW10 ------ SW11 ------ SW12
//        |  Cell10   |  Cell11   |
// row2: SW20 ------ SW21 ------ SW22
//
// 64-bit mesh (4 switches, 1 cell):
//       col0        col1
// row0: SW_A ------ SW_B
//        |  Cell_A   |
// row1: SW_C ------ SW_D
//
// Cell00: pe_addi32  (i32,i32)->i32  (pure 32-bit)
// Cell01: pe_extsi   (i32)->i64      (32-bit in from SW01, 64-bit out to SW_A)
// Cell10: pe_addi32  (i32,i32)->i32  (pure 32-bit)
// Cell11: pe_extsi   (i32)->i64      (32-bit in from SW11, 64-bit out to SW_C)
// Cell_A: pe_addi64  (i64,i64)->i64  (pure 64-bit)
//
// East/South edges: direct.  West/North edges: via FIFO (breaks loops).
//
// Module I/O:
//   inputs:  in0:32 -> SW00, in1:32 -> SW02, in2:32 -> SW20
//   outputs: SW_D -> out0:64

module {
  fabric.pe @pe_addi32(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_extsi(%arg0: !dataflow.bits<32>) -> (!dataflow.bits<64>) {
  ^bb0(%arg0: i32):
    %0 = arith.extsi %arg0 : i32 to i64
    fabric.yield %0 : i64
  }
  fabric.pe @pe_addi64(%arg0: !dataflow.bits<64>, %arg1: !dataflow.bits<64>) -> (!dataflow.bits<64>) {
  ^bb0(%arg0: i64, %arg1: i64):
    %0 = arith.addi %arg0, %arg1 : i64
    fabric.yield %0 : i64
  }

  fabric.module @diff_width_extsi(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>
  ) -> (!dataflow.bits<64>) {

    // ==== 32-bit mesh switches ====

    // SW00 [3in 3out]
    %sw00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in0, %fifo_01_00, %fifo_10_00
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW01 [3in 4out]
    %sw01:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw00#0, %fifo_02_01, %fifo_11_01
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW02 [3in 2out]
    %sw02:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in1, %sw01#1, %fifo_12_02
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // SW10 [3in 5out]
    %sw10:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw00#1, %fifo_11_10, %fifo_20_10
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    // SW11 [5in 5out]
    %sw11:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw01#2, %sw10#1, %fifo_12_11, %fifo_21_11, %pe_cell00
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    // SW12 [3in 3out]
    %sw12:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw02#1, %sw11#2, %fifo_22_12
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW20 [3in 3out]
    %sw20:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in2, %sw10#2, %fifo_21_20
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW21 [4in 3out]
    %sw21:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw11#3, %sw20#1, %fifo_22_21, %pe_cell10
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW22 [2in 2out]
    %sw22:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %sw12#2, %sw21#2
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // ==== 32-bit FIFOs (West and North backward edges) ====

    %fifo_01_00 = fabric.fifo [depth = 2] %sw01#0 : !dataflow.bits<32>
    %fifo_02_01 = fabric.fifo [depth = 2] %sw02#0 : !dataflow.bits<32>
    %fifo_11_10 = fabric.fifo [depth = 2] %sw11#1 : !dataflow.bits<32>
    %fifo_12_11 = fabric.fifo [depth = 2] %sw12#1 : !dataflow.bits<32>
    %fifo_21_20 = fabric.fifo [depth = 2] %sw21#1 : !dataflow.bits<32>
    %fifo_22_21 = fabric.fifo [depth = 2] %sw22#1 : !dataflow.bits<32>

    %fifo_10_00 = fabric.fifo [depth = 2] %sw10#0 : !dataflow.bits<32>
    %fifo_11_01 = fabric.fifo [depth = 2] %sw11#0 : !dataflow.bits<32>
    %fifo_12_02 = fabric.fifo [depth = 2] %sw12#0 : !dataflow.bits<32>
    %fifo_20_10 = fabric.fifo [depth = 2] %sw20#0 : !dataflow.bits<32>
    %fifo_21_11 = fabric.fifo [depth = 2] %sw21#0 : !dataflow.bits<32>
    %fifo_22_12 = fabric.fifo [depth = 2] %sw22#0 : !dataflow.bits<32>

    // ==== 64-bit mesh switches ====

    // SW_A [3in 3out]
    %sw_a:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %pe_cell01, %fifo_b_a, %fifo_c_a
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    // SW_B [2in 2out]
    %sw_b:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %sw_a#0, %fifo_d_b
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    // SW_C [3in 3out]
    %sw_c:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %pe_cell11, %sw_a#1, %fifo_d_c
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    // SW_D [3in 3out]
    %sw_d:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw_b#1, %sw_c#1, %pe_cell_a
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    // ==== 64-bit FIFOs ====

    %fifo_b_a = fabric.fifo [depth = 2] %sw_b#0 : !dataflow.bits<64>
    %fifo_d_b = fabric.fifo [depth = 2] %sw_d#0 : !dataflow.bits<64>
    %fifo_c_a = fabric.fifo [depth = 2] %sw_c#0 : !dataflow.bits<64>
    %fifo_d_c = fabric.fifo [depth = 2] %sw_d#1 : !dataflow.bits<64>

    // ==== PE Instances ====

    // Cell[0,0]: addi32, in0<-SW00(TL), in1<-SW10(BL), out->SW11(BR)
    %pe_cell00 = fabric.instance @pe_addi32(%sw00#2, %sw10#3)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Cell[0,1]: extsi, 32-in<-SW01(TL), 64-out->SW_A (cross-mesh bridge)
    %pe_cell01 = fabric.instance @pe_extsi(%sw01#3)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<32>) -> !dataflow.bits<64>

    // Cell[1,0]: addi32, in0<-SW10(TL), in1<-SW20(BL), out->SW21(BR)
    %pe_cell10 = fabric.instance @pe_addi32(%sw10#4, %sw20#2)
        {sym_name = "pe_1_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Cell[1,1]: extsi, 32-in<-SW11(TL), 64-out->SW_C (cross-mesh bridge)
    %pe_cell11 = fabric.instance @pe_extsi(%sw11#4)
        {sym_name = "pe_1_1"}
        : (!dataflow.bits<32>) -> !dataflow.bits<64>

    // Cell_A: addi64, in0<-SW_A(TL), in1<-SW_C(BL), out->SW_D(BR)
    %pe_cell_a = fabric.instance @pe_addi64(%sw_a#2, %sw_c#2)
        {sym_name = "pe_a_0"}
        : (!dataflow.bits<64>, !dataflow.bits<64>) -> !dataflow.bits<64>

    // Module output: 64-bit result from SW_D
    fabric.yield %sw_d#2 : !dataflow.bits<64>
  }
}
