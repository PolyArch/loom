// Fabric for cmpi + select composition: compare two values, select min.

module {
  fabric.pe @pe_cmpi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>)
      -> (!dataflow.bits<1>) {
  ^bb0(%a: i32, %b: i32):
    %c = arith.cmpi slt, %a, %b : i32
    fabric.yield %c : i1
  }

  fabric.pe @pe_select(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>,
                       %arg2: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%cond: i1, %t: i32, %f: i32):
    %r = arith.select %cond, %t, %f : i32
    fabric.yield %r : i32
  }

  fabric.module @cmp_select(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // Input switches with fanout for both PEs
    %sw_a:3 = fabric.switch [connectivity_table = [
        1, 1, 1]]
        %in0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    %sw_b:3 = fabric.switch [connectivity_table = [
        1, 1, 1]]
        %in1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>

    // cmpi PE: compare a, b -> i1
    %cmpi_out = fabric.instance @pe_cmpi(%sw_a#0, %sw_b#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<1>

    // 1-bit switch to route comparison result
    %sw_cmp:1 = fabric.switch [connectivity_table = [
        1]]
        %cmpi_out
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    // select PE: condition + data -> result
    %sel_out = fabric.instance @pe_select(%sw_cmp#0, %sw_a#1, %sw_b#1)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %sel_out
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
