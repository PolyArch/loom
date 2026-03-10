// Temporal-techmap fabric: 1 temporal PE with a multi-op MAC FU.
// Tests group-candidate tech-mapping on temporal PE FU sub-nodes.
//
// The MAC FU has body: muli(%a, %b) -> addi(result, %c).
// DFGs matching this pattern should be grouped onto a single FU.
//
// Tagged domain: 2x2 switch mesh
//       col0        col1
// row0: ST00 ------ ST01
//        |   TPE     |
// row1: ST10 ------ ST11
//
// TPE (Cell[0,0]): in0=ST00, in1=ST10, in2=ST01, out->ST11
//   (3-input temporal PE for MAC: a, b, c)
//
// Entry: SN -> 3 add_tags -> ST00(in0), ST01(in1/in2), ST10(in2)
// Exit:  ST11 -> del_tag -> fifo -> out0

module {
  // Temporal PE with MAC FU (multi-op body: muli then addi)
  fabric.temporal_pe @tpe_mac(
      %arg0: !dataflow.tagged<!dataflow.bits<32>, i4>,
      %arg1: !dataflow.tagged<!dataflow.bits<32>, i4>,
      %arg2: !dataflow.tagged<!dataflow.bits<32>, i4>
  ) [num_register = 0, num_instruction = 4, reg_fifo_depth = 0]
    -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
    fabric.pe @fu_mac(%a: i32, %b: i32, %c: i32) -> (i32) {
      %0 = arith.muli %a, %b : i32
      %1 = arith.addi %0, %c : i32
      fabric.yield %1 : i32
    }
    fabric.yield
  }

  fabric.module @temporal_techmap(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // ======== Native Input Switch ========
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

    // ======== Tagged Switches (2x2 mesh) ========

    // ST00 [3in, 3out]
    //   in:  add_tag0, fifo<-ST01(W), fifo<-ST10(N)
    //   out: #0 E->ST01, #1 S->ST10, #2 TPE in0
    %st00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %at0, %fifo_st01_st00, %fifo_st10_st00
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST01 [3in, 3out]
    //   in:  ST00#0(E), fifo<-ST11(N), add_tag1
    //   out: #0 W->fifo->ST00, #1 S->ST11, #2 TPE in2
    %st01:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st00#0, %fifo_st11_st01, %at1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST10 [3in, 3out]
    //   in:  ST00#1(S), fifo<-ST11(W), add_tag2
    //   out: #0 N->fifo->ST00, #1 E->ST11, #2 TPE in1
    %st10:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st00#1, %fifo_st11_st10, %at2
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST11 [3in, 3out]
    //   in:  ST01#1(S), ST10#1(E), TPE out
    //   out: #0 N->fifo->ST01, #1 W->fifo->ST10, #2 del_tag
    %st11:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st01#1, %st10#1, %tpe
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Backward FIFOs ========
    %fifo_st01_st00 = fabric.fifo [depth = 2] %st01#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st11_st10 = fabric.fifo [depth = 2] %st11#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st10_st00 = fabric.fifo [depth = 2] %st10#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st11_st01 = fabric.fifo [depth = 2] %st11#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== TPE Instance ========
    // TPE (Cell[0,0]): TL=ST00, BL=ST10, TR=ST01, output->ST11(BR)
    %tpe = fabric.instance @tpe_mac(%st00#2, %st10#2, %st01#2)
        {sym_name = "tpe_0"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Del Tag + FIFO (tagged -> native) ========
    %dt = fabric.del_tag %st11#2
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %fifo_out = fabric.fifo [depth = 2] %dt : !dataflow.bits<32>

    // ======== Module Output ========
    fabric.yield %fifo_out : !dataflow.bits<32>
  }
}
