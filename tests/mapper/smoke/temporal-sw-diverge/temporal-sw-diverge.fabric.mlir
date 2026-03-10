// Temporal-sw-diverge fabric: per-tag route divergence test.
// 1 temporal_sw (1 input, 2 outputs) diverges streams by tag.
// 2 TPEs (fu_add each, 1 slot) receive the diverged streams.
//
// Data flow:
//   Stream A (tag=0): in0 -> at0 -> s_merge -> tsw.in0 -> tsw.out0 -> sa -> TPE_A
//   Stream B (tag=1): in2 -> at2 -> s_merge -> tsw.in0 -> tsw.out1 -> sb -> TPE_B
//   Operand A side: in1 -> at1 -> sa -> TPE_A (tag=0 second operand)
//   Operand B side: in3 -> at3 -> sb -> TPE_B (tag=1 second operand)

module {
  // Temporal PE: add FU only, 1 instruction slot
  fabric.temporal_pe @tpe_add1(
      %arg0: !dataflow.tagged<!dataflow.bits<32>, i4>,
      %arg1: !dataflow.tagged<!dataflow.bits<32>, i4>
  ) [num_register = 0, num_instruction = 1, reg_fifo_depth = 0]
    -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
    fabric.pe @fu_add(%a: i32, %b: i32) -> (i32) {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @temporal_sw_diverge(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>,
      %in3: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>, !dataflow.bits<32>) {

    // ======== Native Input Switch ========
    %sn:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1,
        1, 1, 1, 1]]
        %in0, %in1, %in2, %in3
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>,
          !dataflow.bits<32>, !dataflow.bits<32>

    // ======== Add Tags (native -> tagged) ========
    %at0 = fabric.add_tag %sn#0 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %at1 = fabric.add_tag %sn#1 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %at2 = fabric.add_tag %sn#2 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %at3 = fabric.add_tag %sn#3 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Merge Switch: combines streams for temporal_sw ========
    // at0 (tag=0) and at2 (tag=1) merge into one port
    %s_merge:1 = fabric.switch [connectivity_table = [
        1, 1]]
        %at0, %at2
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Temporal Switch: diverge by tag ========
    // in0 carries both tag=0 and tag=1 data
    // out0 -> tag=0 stream (to TPE_A)
    // out1 -> tag=1 stream (to TPE_B)
    %tsw:2 = fabric.temporal_sw [num_route_table = 2,
        connectivity_table = [1, 1]]
        %s_merge#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== TPE_A Input Switch ========
    %sa:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %tsw#0, %at1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== TPE_B Input Switch ========
    %sb:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %tsw#1, %at3
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== TPE Instances ========
    %tpe_a = fabric.instance @tpe_add1(%sa#0, %sa#1)
        {sym_name = "tpe_a"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    %tpe_b = fabric.instance @tpe_add1(%sb#0, %sb#1)
        {sym_name = "tpe_b"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Del Tags + FIFOs (tagged -> native) ========
    %dt_a = fabric.del_tag %tpe_a
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %fifo_a = fabric.fifo [depth = 2] %dt_a : !dataflow.bits<32>

    %dt_b = fabric.del_tag %tpe_b
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %fifo_b = fabric.fifo [depth = 2] %dt_b : !dataflow.bits<32>

    // ======== Module Outputs ========
    fabric.yield %fifo_a, %fifo_b : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
