// Temporal-only fabric: 2 temporal PEs, NO native PEs.
// Tests that temporal PE FU mapping works as the sole compute resource.
//
// Topology:
//   Native input switch SN distributes 3 inputs to 3 add_tags.
//   Tagged domain: 3-row x 2-col switch mesh.
//
//       col0        col1
// row0: ST00 ------ ST01
//        |  TPE_A    |
// row1: ST10 ------ ST11
//        |  TPE_B    |
// row2: ST20 ------ ST21
//
// TPE_A (Cell[0,0]): in0=ST00, in1=ST10, out->ST11
// TPE_B (Cell[1,0]): in0=ST10, in1=ST20, out->ST21
//
// Entry: SN -> add_tag0 -> ST00, add_tag1 -> ST01, add_tag2 -> ST20
// Exit:  ST21 -> del_tag -> fifo -> out0
//
// East/South edges: direct.  West/North edges: via FIFO.

module {
  // Temporal PE definition: add + mul FUs, 4 instruction slots
  fabric.temporal_pe @tpe_alu(
      %arg0: !dataflow.tagged<!dataflow.bits<32>, i4>,
      %arg1: !dataflow.tagged<!dataflow.bits<32>, i4>
  ) [num_register = 0, num_instruction = 4, reg_fifo_depth = 0]
    -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
    fabric.pe @fu_add(%a: i32, %b: i32) -> (i32) {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.pe @fu_mul(%a: i32, %b: i32) -> (i32) {
      %r = arith.muli %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @temporal_only(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // ======== Native Input Switch ========
    // SN [3in, 3out]: distributes module inputs to add_tags
    %sn:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in0, %in1, %in2
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // ======== Add Tags (native -> tagged) ========
    %at0 = fabric.add_tag %sn#0 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %at1 = fabric.add_tag %sn#1 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %at2 = fabric.add_tag %sn#2 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Tagged Switches (3 rows x 2 cols) ========

    // ST00 [3in, 3out]
    //   in:  add_tag0, fifo<-ST01(W), fifo<-ST10(N)
    //   out: #0 E->ST01, #1 S->ST10, #2 TPE_A in0
    %st00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %at0, %fifo_st01_st00, %fifo_st10_st00
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST01 [3in, 2out]
    //   in:  ST00#0(E), fifo<-ST11(N), add_tag1
    //   out: #0 W->fifo->ST00, #1 S->ST11
    %st01:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %st00#0, %fifo_st11_st01, %at1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST10 [3in, 5out]
    //   in:  ST00#1(S), fifo<-ST11(W), fifo<-ST20(N)
    //   out: #0 N->fifo->ST00, #1 E->ST11, #2 S->ST20,
    //        #3 TPE_A in1, #4 TPE_B in0
    %st10:5 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st00#1, %fifo_st11_st10, %fifo_st20_st10
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST11 [4in, 3out]
    //   in:  ST01#1(S), ST10#1(E), fifo<-ST21(N), TPE_A out
    //   out: #0 N->fifo->ST01, #1 W->fifo->ST10, #2 S->ST21
    %st11:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st01#1, %st10#1, %fifo_st21_st11, %tpe_a
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST20 [3in, 3out]
    //   in:  ST10#2(S), fifo<-ST21(W), add_tag2
    //   out: #0 N->fifo->ST10, #1 E->ST21, #2 TPE_B in1
    %st20:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st10#2, %fifo_st21_st20, %at2
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST21 [3in, 3out]
    //   in:  ST11#2(S), ST20#1(E), TPE_B out
    //   out: #0 N->fifo->ST11, #1 W->fifo->ST20, #2 del_tag
    %st21:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st11#2, %st20#1, %tpe_b
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Backward FIFOs (West and North) ========

    // West FIFOs
    %fifo_st01_st00 = fabric.fifo [depth = 2] %st01#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st11_st10 = fabric.fifo [depth = 2] %st11#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st21_st20 = fabric.fifo [depth = 2] %st21#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>

    // North FIFOs
    %fifo_st10_st00 = fabric.fifo [depth = 2] %st10#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st11_st01 = fabric.fifo [depth = 2] %st11#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st20_st10 = fabric.fifo [depth = 2] %st20#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st21_st11 = fabric.fifo [depth = 2] %st21#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== TPE Instances ========

    // TPE_A (Cell[0,0]): TL=ST00, BL=ST10, output->ST11(BR)
    %tpe_a = fabric.instance @tpe_alu(%st00#2, %st10#3)
        {sym_name = "tpe_a"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // TPE_B (Cell[1,0]): TL=ST10, BL=ST20, output->ST21(BR)
    %tpe_b = fabric.instance @tpe_alu(%st10#4, %st20#2)
        {sym_name = "tpe_b"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Del Tag + FIFO (tagged -> native) ========
    %dt = fabric.del_tag %st21#2
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %fifo_out = fabric.fifo [depth = 2] %dt : !dataflow.bits<32>

    // ======== Module Output ========
    fabric.yield %fifo_out : !dataflow.bits<32>
  }
}
