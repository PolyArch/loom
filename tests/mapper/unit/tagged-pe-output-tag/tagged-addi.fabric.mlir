// Tagged PE ADG with output_tag: tests non-temporal tagged PE mapping.
// Topology: native input switch -> add_tags -> tagged 2x2 mesh -> del_tag -> output
//
// The PE definition uses tagged ports but native body types.
// The mapper must match native DFG ops to tagged PEs via tag-unwrapping
// and emit output_tag in configured MLIR.

module {
  fabric.pe @pe_tagged_addi(
      %arg0: !dataflow.tagged<!dataflow.bits<32>, i4>,
      %arg1: !dataflow.tagged<!dataflow.bits<32>, i4>
  ) {output_tag = [0 : i4]} -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  ^bb0(%a: i32, %b: i32):
    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }

  fabric.module @tagged_addi(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // Native input switch distributes module inputs to add_tags.
    %sn:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %in0, %in1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // Native -> tagged domain crossing.
    %at0 = fabric.add_tag %sn#0 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>
    %at1 = fabric.add_tag %sn#1 {tag = 0 : i4}
        : !dataflow.bits<32> -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // Tagged 2x2 switch mesh.
    // ST00: inputs from add_tag0, fifo_st01, fifo_st10
    %st00:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %at0, %fifo_st01_st00, %fifo_st10_st00
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST01: inputs from ST00#0, add_tag1, fifo_st11
    %st01:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st00#0, %at1, %fifo_st11_st01
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST10: inputs from ST00#1, pe_out0, fifo_st11
    %st10:3 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st00#1, %pe_out0, %fifo_st11_st10
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ST11: inputs from ST01#1, ST10#1, pe_out1; 4 outputs.
    %st11:4 = fabric.switch [connectivity_table = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        %st01#1, %st10#1, %pe_out1
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // Backward FIFOs.
    %fifo_st01_st00 = fabric.fifo [depth = 2] %st01#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st10_st00 = fabric.fifo [depth = 2] %st10#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st11_st01 = fabric.fifo [depth = 2] %st11#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
    %fifo_st11_st10 = fabric.fifo [depth = 2] %st11#1
        : !dataflow.tagged<!dataflow.bits<32>, i4>

    // Tagged PE instances with output_tag attribute.
    %pe_out0 = fabric.instance @pe_tagged_addi(%st00#2, %st01#2)
        {sym_name = "pe_0", output_tag = [0 : i4]}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    %pe_out1 = fabric.instance @pe_tagged_addi(%st10#2, %st11#2)
        {sym_name = "pe_1", output_tag = [0 : i4]}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // Tagged -> native domain crossing.
    %dt = fabric.del_tag %st11#3
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %fifo_out = fabric.fifo [depth = 2] %dt : !dataflow.bits<32>

    fabric.yield %fifo_out : !dataflow.bits<32>
  }
}
