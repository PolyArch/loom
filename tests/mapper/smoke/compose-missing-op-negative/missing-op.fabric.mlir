// Fabric with only addi PEs (no muli). DFG requires muli -> should fail.

module {
  fabric.pe @pe_addi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
    %0 = arith.addi %a, %b : i32
    fabric.yield %0 : i32
  }

  fabric.module @missing_op(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    %sw0:1 = fabric.switch [connectivity_table = [
        1]]
        %in0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw1:1 = fabric.switch [connectivity_table = [
        1]]
        %in1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %pe0 = fabric.instance @pe_addi(%sw0#0, %sw1#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
