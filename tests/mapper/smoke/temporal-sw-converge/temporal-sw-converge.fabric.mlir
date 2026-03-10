// Temporal-sw-converge fabric: per-tag route convergence test.
// 1 temporal_sw (2 inputs, 1 output) converges streams to same output.
// 1 TPE (fu_add, 2 slots) processes both streams.
//
// Data flow:
//   Stream A (tag=0): in0 -> at0 -> tsw.in0 -> tsw.out0 -> st -> TPE
//   Stream B (tag=1): in2 -> at2 -> tsw.in1 -> tsw.out0 -> st -> TPE
//   Operand merge:    in1 -> at1, in3 -> at3 -> s_merge -> st -> TPE

module {
  // Temporal PE: add FU only, 2 instruction slots
  fabric.temporal_pe @tpe_add2(
      %arg0: !dataflow.tagged<!dataflow.bits<32>, i4>,
      %arg1: !dataflow.tagged<!dataflow.bits<32>, i4>
  ) [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
    -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
    fabric.pe @fu_add(%a: i32, %b: i32) -> (i32) {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @temporal_sw_converge(
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

    // ======== Temporal Switch: converge to same output ========
    // in0 carries tag=0, in1 carries tag=1
    // Both route to out0 (convergence)
    %tsw:1 = fabric.temporal_sw [num_route_table = 2,
        connectivity_table = [1, 1]]
        %at0, %at2
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Merge Switch: second operands ========
    %s_merge:1 = fabric.switch [connectivity_table = [
        1, 1]]
        %at1, %at3
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== TPE Input Switch ========
    %st:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %tsw#0, %s_merge#0
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== TPE Instance ========
    %tpe = fabric.instance @tpe_add2(%st#0, %st#1)
        {sym_name = "tpe_0"}
        : (!dataflow.tagged<!dataflow.bits<32>, i4>,
           !dataflow.tagged<!dataflow.bits<32>, i4>)
          -> !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Output Switch ========
    %s_out:2 = fabric.switch [connectivity_table = [
        1, 1]]
        %tpe
        : !dataflow.tagged<!dataflow.bits<32>, i4>
       -> !dataflow.tagged<!dataflow.bits<32>, i4>,
          !dataflow.tagged<!dataflow.bits<32>, i4>

    // ======== Del Tags + FIFOs (tagged -> native) ========
    %dt0 = fabric.del_tag %s_out#0
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %fifo0 = fabric.fifo [depth = 2] %dt0 : !dataflow.bits<32>

    %dt1 = fabric.del_tag %s_out#1
        : !dataflow.tagged<!dataflow.bits<32>, i4> -> !dataflow.bits<32>
    %fifo1 = fabric.fifo [depth = 2] %dt1 : !dataflow.bits<32>

    // ======== Module Outputs ========
    fabric.yield %fifo0, %fifo1 : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
