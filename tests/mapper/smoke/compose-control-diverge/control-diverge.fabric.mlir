// Fabric for cond_br + addi/subi: branch then compute on each path.

module {
  fabric.pe @pe_cond_br(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>)
      -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  ^bb0(%cond: i1, %val: i32):
    %t, %f = handshake.cond_br %cond, %val : i32
    fabric.yield %t, %f : i32, i32
  }

  fabric.pe @pe_addi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>)
      -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
    %0 = arith.addi %a, %b : i32
    fabric.yield %0 : i32
  }

  fabric.pe @pe_subi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>)
      -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
    %0 = arith.subi %a, %b : i32
    fabric.yield %0 : i32
  }

  fabric.module @control_diverge(
      %cond_in: !dataflow.bits<1>,
      %a_in: !dataflow.bits<32>,
      %b_in: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>, !dataflow.bits<32>) {

    // 1-bit switch for condition
    %sw_c:1 = fabric.switch [connectivity_table = [
        1]]
        %cond_in
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    // 32-bit input switches with fanout
    %sw_a:1 = fabric.switch [connectivity_table = [
        1]]
        %a_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_b:3 = fabric.switch [connectivity_table = [
        1, 1, 1]]
        %b_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // cond_br PE: split data based on condition
    %br:2 = fabric.instance @pe_cond_br(%sw_c#0, %sw_a#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<1>, !dataflow.bits<32>)
       -> (!dataflow.bits<32>, !dataflow.bits<32>)

    // Route true/false outputs
    %sw_t:1 = fabric.switch [connectivity_table = [
        1]]
        %br#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_f:1 = fabric.switch [connectivity_table = [
        1]]
        %br#1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // addi on true path
    %add_out = fabric.instance @pe_addi(%sw_t#0, %sw_b#0)
        {sym_name = "pe_1_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // subi on false path
    %sub_out = fabric.instance @pe_subi(%sw_f#0, %sw_b#1)
        {sym_name = "pe_1_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Output switches
    %sw_out_t:1 = fabric.switch [connectivity_table = [
        1]]
        %add_out
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_out_f:1 = fabric.switch [connectivity_table = [
        1]]
        %sub_out
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out_t#0, %sw_out_f#0 : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
