// Temporal-width-64 fabric: 64-bit width plane temporal mapping.
// 1 TPE with fu_add + fu_mul, num_instruction=4.
// Tagged domain: bits<64> with i4 tags.
//
// 2x2 switch mesh:
//       col0        col1
// row0: ST00 ------ ST01
//        |   TPE     |
// row1: ST10 ------ ST11
//
// TPE (Cell[0,0]): in0=ST00, in1=ST10, out->ST11
//
// Entry: SN -> 3 add_tags -> ST00(in0), ST01(in1), ST10(in2)
// Exit:  ST11 -> del_tag -> fifo -> out0

module {
  // Temporal PE: add + mul FUs, 4 instruction slots, 64-bit data
  fabric.temporal_pe @tpe_alu64(
      %arg0: !dataflow.tagged<!dataflow.bits<64>, i4>,
      %arg1: !dataflow.tagged<!dataflow.bits<64>, i4>
  ) [num_register = 0, num_instruction = 4, reg_fifo_depth = 0]
    -> (!dataflow.tagged<!dataflow.bits<64>, i4>) {
    fabric.pe @fu_add(%a: i64, %b: i64) -> (i64) {
      %r = arith.addi %a, %b : i64
      fabric.yield %r : i64
    }
    fabric.pe @fu_mul(%a: i64, %b: i64) -> (i64) {
      %r = arith.muli %a, %b : i64
      fabric.yield %r : i64
    }
    fabric.yield
  }

  fabric.module @temporal_width_64(
      %in0: !dataflow.bits<64>,
      %in1: !dataflow.bits<64>,
      %in2: !dataflow.bits<64>
  ) -> (!dataflow.bits<64>) {

    // ======== Native Input Switch ========
    %sn:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %in0, %in1, %in2
        : !dataflow.bits<64>
       -> !dataflow.bits<64>, !dataflow.bits<64>, !dataflow.bits<64>

    // ======== Add Tags (native -> tagged) ========
    %at0 = fabric.add_tag %sn#0 {tag = 0 : i4}
        : !dataflow.bits<64> -> !dataflow.tagged<!dataflow.bits<64>, i4>
    %at1 = fabric.add_tag %sn#1 {tag = 0 : i4}
        : !dataflow.bits<64> -> !dataflow.tagged<!dataflow.bits<64>, i4>
    %at2 = fabric.add_tag %sn#2 {tag = 0 : i4}
        : !dataflow.bits<64> -> !dataflow.tagged<!dataflow.bits<64>, i4>

    // ======== Tagged Switches (2x2 mesh) ========

    // ST00 [3in, 3out]
    %st00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %at0, %fifo_st01_st00, %fifo_st10_st00
        : !dataflow.tagged<!dataflow.bits<64>, i4>
       -> !dataflow.tagged<!dataflow.bits<64>, i4>,
          !dataflow.tagged<!dataflow.bits<64>, i4>,
          !dataflow.tagged<!dataflow.bits<64>, i4>

    // ST01 [3in, 2out]
    %st01:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1]]
        %st00#0, %fifo_st11_st01, %at1
        : !dataflow.tagged<!dataflow.bits<64>, i4>
       -> !dataflow.tagged<!dataflow.bits<64>, i4>,
          !dataflow.tagged<!dataflow.bits<64>, i4>

    // ST10 [3in, 3out]
    %st10:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st00#1, %fifo_st11_st10, %at2
        : !dataflow.tagged<!dataflow.bits<64>, i4>
       -> !dataflow.tagged<!dataflow.bits<64>, i4>,
          !dataflow.tagged<!dataflow.bits<64>, i4>,
          !dataflow.tagged<!dataflow.bits<64>, i4>

    // ST11 [3in, 3out]
    %st11:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st01#1, %st10#1, %tpe
        : !dataflow.tagged<!dataflow.bits<64>, i4>
       -> !dataflow.tagged<!dataflow.bits<64>, i4>,
          !dataflow.tagged<!dataflow.bits<64>, i4>,
          !dataflow.tagged<!dataflow.bits<64>, i4>

    // ======== Backward FIFOs ========
    %fifo_st01_st00 = fabric.fifo [depth = 2] %st01#0
        : !dataflow.tagged<!dataflow.bits<64>, i4>
    %fifo_st11_st10 = fabric.fifo [depth = 2] %st11#1
        : !dataflow.tagged<!dataflow.bits<64>, i4>
    %fifo_st10_st00 = fabric.fifo [depth = 2] %st10#0
        : !dataflow.tagged<!dataflow.bits<64>, i4>
    %fifo_st11_st01 = fabric.fifo [depth = 2] %st11#0
        : !dataflow.tagged<!dataflow.bits<64>, i4>

    // ======== TPE Instance ========
    %tpe = fabric.instance @tpe_alu64(%st00#2, %st10#2)
        {sym_name = "tpe_0"}
        : (!dataflow.tagged<!dataflow.bits<64>, i4>,
           !dataflow.tagged<!dataflow.bits<64>, i4>)
          -> !dataflow.tagged<!dataflow.bits<64>, i4>

    // ======== Del Tag + FIFO (tagged -> native) ========
    %dt = fabric.del_tag %st11#2
        : !dataflow.tagged<!dataflow.bits<64>, i4> -> !dataflow.bits<64>
    %fifo_out = fabric.fifo [depth = 2] %dt : !dataflow.bits<64>

    // ======== Module Output ========
    fabric.yield %fifo_out : !dataflow.bits<64>
  }
}
