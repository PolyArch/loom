// ADG: Techmap + diff-width combined test.
// 3x3 32-bit mesh + 2x2 64-bit mesh with composite width-changing PEs.
//
// 32-bit mesh (9 switches, 4 cells):
//       col0        col1        col2
// row0: SW00 ------ SW01 ------ SW02
//        |  Cell00   |  Cell01   |
// row1: SW10 ------ SW11 ------ SW12
//        |           |           |
// row2: SW20 ------ SW21 ------ SW22
//
// 64-bit mesh (4 switches, 1 cell):
//       col0        col1
// row0: SW_A ------ SW_B
//        |  Cell_A   |
// row1: SW_C ------ SW_D
//
// Cell00: pe_add_ext        (i32,i32)->i64  body: addi32->extsi (techmap, bridge 32->64)
// Cell01: pe_add_ext        (i32,i32)->i64  body: addi32->extsi (techmap, bridge 32->64)
// Cell10/11: empty (routing only)
// Cell_A: pe_add_trunc_ext  (i64,i64)->i64  body: addi64->trunci->extsi {64->32->64}
//
// East/South edges: direct.  West/North edges: via FIFO (breaks loops).
//
// Module I/O:
//   inputs:  in0:32 -> SW00, in1:32 -> SW02, in2:32 -> SW20, in3:32 -> SW22,
//            in4:64 -> SW_B, in5:64 -> SW_D
//   outputs: out0:64 from SW_D

module {
  // Composite PE: addi32 -> extsi.  (i32, i32) -> i64
  // Adds in 32-bit then sign-extends to 64-bit.  Linear chain body.
  fabric.pe @pe_add_ext(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<64>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.extsi %0 : i32 to i64
    fabric.yield %1 : i64
  }

  // Composite PE: addi64 -> trunci -> extsi.  (i64, i64) -> i64
  // The {(64,64) -> 32 -> 64} pattern: adds in 64-bit, narrows to 32, widens back.
  fabric.pe @pe_add_trunc_ext(%arg0: !dataflow.bits<64>, %arg1: !dataflow.bits<64>) -> (!dataflow.bits<64>) {
  ^bb0(%arg0: i64, %arg1: i64):
    %0 = arith.addi %arg0, %arg1 : i64
    %1 = arith.trunci %0 : i64 to i32
    %2 = arith.extsi %1 : i32 to i64
    fabric.yield %2 : i64
  }

  fabric.module @techmap_diffwidth(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>,
      %in3: !dataflow.bits<32>,
      %in4: !dataflow.bits<64>,
      %in5: !dataflow.bits<64>
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

    // SW10 [3in 4out]
    %sw10:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw00#1, %fifo_11_10, %fifo_20_10
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>

    // SW11 [4in 5out]
    %sw11:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw01#2, %sw10#1, %fifo_12_11, %fifo_21_11
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    // SW12 [3in 3out]
    %sw12:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw02#1, %sw11#2, %fifo_22_12
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW20 [3in 2out]
    %sw20:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in2, %sw10#2, %fifo_21_20
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // SW21 [3in 3out]
    %sw21:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw11#3, %sw20#1, %fifo_22_21
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW22 [3in 2out]
    %sw22:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in3, %sw12#2, %sw21#2
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // ==== 32-bit FIFOs ====

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
        %pe_cell00, %fifo_b_a, %fifo_c_a
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    // SW_B [3in 2out]
    %sw_b:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %in4, %sw_a#0, %fifo_d_b
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>

    // SW_C [3in 3out]
    %sw_c:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %pe_cell01, %sw_a#1, %fifo_d_c
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    // SW_D [4in 3out]
    %sw_d:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in5, %sw_b#1, %sw_c#1, %pe_cell_a
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    // ==== 64-bit FIFOs ====

    %fifo_b_a = fabric.fifo [depth = 2] %sw_b#0 : !dataflow.bits<64>
    %fifo_d_b = fabric.fifo [depth = 2] %sw_d#0 : !dataflow.bits<64>
    %fifo_c_a = fabric.fifo [depth = 2] %sw_c#0 : !dataflow.bits<64>
    %fifo_d_c = fabric.fifo [depth = 2] %sw_d#1 : !dataflow.bits<64>

    // ==== PE Instances ====

    // Cell[0,0]: pe_add_ext (techmap), in0<-SW00(TL), in1<-SW10(BL), 64-out->SW_A
    %pe_cell00 = fabric.instance @pe_add_ext(%sw00#2, %sw10#3)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<64>

    // Cell[0,1]: pe_add_ext (techmap), in0<-SW01(TL), in1<-SW11(BL), 64-out->SW_C
    %pe_cell01 = fabric.instance @pe_add_ext(%sw01#3, %sw11#4)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<64>

    // Cell_A: pe_add_trunc_ext (techmap), in0<-SW_A(TL), in1<-SW_C(BL), out->SW_D(BR)
    %pe_cell_a = fabric.instance @pe_add_trunc_ext(%sw_a#2, %sw_c#2)
        {sym_name = "pe_a_0"}
        : (!dataflow.bits<64>, !dataflow.bits<64>) -> !dataflow.bits<64>

    // Module output: 64-bit from SW_D
    fabric.yield %sw_d#2 : !dataflow.bits<64>
  }
}
