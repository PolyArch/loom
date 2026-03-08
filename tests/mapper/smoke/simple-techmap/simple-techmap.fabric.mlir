// 3x3 switch lattice mesh with 4 heterogeneous PEs for technology mapping.
//
//       col0        col1        col2
// row0: SW00 ------ SW01 ------ SW02
//        |  Cell00   |  Cell01   |
// row1: SW10 ------ SW11 ------ SW12
//        |  Cell10   |  Cell11   |
// row2: SW20 ------ SW21 ------ SW22
//
// Cell00: pe_addi     (i32 add, 2-in 1-out)
// Cell01: pe_add_mul  ((a+b)*c,  3-in 1-out)
// Cell10: pe_mul_add  ((a*b)+c,  3-in 1-out)
// Cell11: pe_muli     (i32 mul, 2-in 1-out)
//
// PE connections per cell:
//   2-in PE:  input 0 <- TL switch, input 1 <- BL switch
//             output 0 -> BR switch
//   3-in PE:  input 0 <- TL switch, input 1 <- BL switch, input 2 <- TR switch
//             output 0 -> BR switch
//
// All inter-switch edges are direct (no FIFOs, combinational loops present).
//
// Module I/O:
//   inputs:  in0 -> SW00, in1 -> SW02, in2 -> SW20, in3 -> SW22
//   outputs: SW22 -> out0, SW12 -> out1

module {
  fabric.pe @pe_addi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_muli(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.muli %arg0, %arg1 : i32
    fabric.yield %0 : i32
  }
  fabric.pe @pe_add_mul(
      %arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>, %arg2: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    %1 = arith.muli %0, %arg2 : i32
    fabric.yield %1 : i32
  }
  fabric.pe @pe_mul_add(
      %arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>, %arg2: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {
  ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):
    %0 = arith.muli %arg0, %arg1 : i32
    %1 = arith.addi %0, %arg2 : i32
    fabric.yield %1 : i32
  }

  fabric.module @simple_techmap(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>,
      %in3: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>, !dataflow.bits<32>) {

    // ---- Switches ----

    // SW00 [3in 3out]
    //   in:  module %in0, SW01(W), SW10(N)
    //   out: #0 E->SW01, #1 S->SW10, #2 PE cell00 in0
    %sw00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in0, %sw01#0, %sw10#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW01 [3in 4out]
    //   in:  SW00#0(E), SW02(W), SW11(N)
    //   out: #0 W->SW00, #1 E->SW02, #2 S->SW11, #3 PE cell01 in0
    %sw01:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw00#0, %sw02#0, %sw11#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW02 [3in 3out]
    //   in:  module %in1, SW01#1(E), SW12(N)
    //   out: #0 W->SW01, #1 S->SW12, #2 PE cell01 in2
    %sw02:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in1, %sw01#1, %sw12#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW10 [3in 5out]
    //   in:  SW00#1(S), SW11(W), SW20(N)
    //   out: #0 N->SW00, #1 E->SW11, #2 S->SW20,
    //        #3 PE cell00 in1, #4 PE cell10 in0
    %sw10:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw00#1, %sw11#1, %sw20#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    // SW11 [5in 7out]  (center switch, most connections)
    //   in:  SW01#2(S), SW10#1(E), SW12(W), SW21(N), PE cell00 out
    //   out: #0 N->SW01, #1 W->SW10, #2 E->SW12, #3 S->SW21,
    //        #4 PE cell01 in1, #5 PE cell10 in2, #6 PE cell11 in0
    %sw11:7 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1]]
        %sw01#2, %sw10#1, %sw12#1, %sw21#0, %pe_cell00
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>

    // SW12 [4in 4out]
    //   in:  SW02#1(S), SW11#2(E), SW22(N), PE cell01 out
    //   out: #0 N->SW02, #1 W->SW11, #2 S->SW22, #3 module out1
    %sw12:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw02#1, %sw11#2, %sw22#0, %pe_cell01
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW20 [3in 3out]
    //   in:  module %in2, SW10#2(S), SW21(W)
    //   out: #0 N->SW10, #1 E->SW21, #2 PE cell10 in1
    %sw20:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in2, %sw10#2, %sw21#1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW21 [4in 4out]
    //   in:  SW11#3(S), SW20#1(E), SW22(W), PE cell10 out
    //   out: #0 N->SW11, #1 W->SW20, #2 E->SW22, #3 PE cell11 in1
    %sw21:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %sw11#3, %sw20#1, %sw22#1, %pe_cell10
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // SW22 [4in 3out]
    //   in:  module %in3, SW12#2(S), SW21#2(E), PE cell11 out
    //   out: #0 N->SW12, #1 W->SW21, #2 module out0
    %sw22:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in3, %sw12#2, %sw21#2, %pe_cell11
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // ---- PE Instances ----

    // Cell[0,0]: addi (2-in 1-out), TL=SW00 BL=SW10, output->SW11(BR)
    %pe_cell00 = fabric.instance @pe_addi(%sw00#2, %sw10#3)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Cell[0,1]: add_mul (3-in 1-out), TL=SW01 BL=SW11 TR=SW02, output->SW12(BR)
    %pe_cell01 = fabric.instance @pe_add_mul(%sw01#3, %sw11#4, %sw02#2)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Cell[1,0]: mul_add (3-in 1-out), TL=SW10 BL=SW20 TR=SW11, output->SW21(BR)
    %pe_cell10 = fabric.instance @pe_mul_add(%sw10#4, %sw20#2, %sw11#5)
        {sym_name = "pe_1_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Cell[1,1]: muli (2-in 1-out), TL=SW11 BL=SW21, output->SW22(BR)
    %pe_cell11 = fabric.instance @pe_muli(%sw11#6, %sw21#3)
        {sym_name = "pe_1_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Module outputs: out0 from SW22 (bottom-right), out1 from SW12 (right-middle)
    fabric.yield %sw22#2, %sw12#3 : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
