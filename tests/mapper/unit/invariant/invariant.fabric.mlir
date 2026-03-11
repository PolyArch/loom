// Fabric with dataflow.invariant PE (exclusive dataflow body).

module {
  fabric.pe @pe_invariant(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>)
      -> (!dataflow.bits<32>) {
  ^bb0(%d: i1, %a: i32):
    %inv = dataflow.invariant %d, %a : i1, i32 -> i32
    fabric.yield %inv : i32
  }

  fabric.module @invariant(
      %ctrl_in: !dataflow.bits<1>,
      %data_in: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // 1-bit switch for control routing
    %sw_c:1 = fabric.switch [connectivity_table = [
        1]]
        %ctrl_in
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    // 32-bit switch for data routing
    %sw_d:1 = fabric.switch [connectivity_table = [
        1]]
        %data_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // invariant PE
    %pe0 = fabric.instance @pe_invariant(%sw_c#0, %sw_d#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<1>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // 32-bit output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
