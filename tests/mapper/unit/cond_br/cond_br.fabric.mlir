// Fabric with cond_br PE (partial-consume, exclusive group).

module {
  fabric.pe @pe_cond_br(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>)
      -> (!dataflow.bits<32>, !dataflow.bits<32>) {
  ^bb0(%cond: i1, %val: i32):
    %t, %f = handshake.cond_br %cond, %val : i32
    fabric.yield %t, %f : i32, i32
  }

  fabric.module @cond_br(
      %cond_in: !dataflow.bits<1>,
      %data_in: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>, !dataflow.bits<32>) {

    // 1-bit switch for condition routing
    %sw_c:1 = fabric.switch [connectivity_table = [
        1]]
        %cond_in
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    // 32-bit switch for data routing
    %sw_d:1 = fabric.switch [connectivity_table = [
        1]]
        %data_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // cond_br PE (1-bit condition + 32-bit data -> 2 x 32-bit outputs)
    %pe0:2 = fabric.instance @pe_cond_br(%sw_c#0, %sw_d#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<1>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)

    // 32-bit output switches
    %sw_out0:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_out1:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out0#0, %sw_out1#0 : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
