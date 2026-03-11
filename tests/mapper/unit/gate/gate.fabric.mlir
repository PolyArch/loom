// Fabric with dataflow.gate PE (exclusive dataflow body).

module {
  fabric.pe @pe_gate(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<1>)
      -> (!dataflow.bits<32>, !dataflow.bits<1>) {
  ^bb0(%val: i32, %cond: i1):
    %afterVal, %afterCond = dataflow.gate %val, %cond : i32, i1 -> i32, i1
    fabric.yield %afterVal, %afterCond : i32, i1
  }

  fabric.module @gate(
      %data_in: !dataflow.bits<32>,
      %ctrl_in: !dataflow.bits<1>
  ) -> (!dataflow.bits<32>, !dataflow.bits<1>) {

    // 32-bit switch for data routing
    %sw_d:1 = fabric.switch [connectivity_table = [
        1]]
        %data_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // 1-bit switch for control routing
    %sw_c:1 = fabric.switch [connectivity_table = [
        1]]
        %ctrl_in
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    // gate PE (2 outputs)
    %pe0:2 = fabric.instance @pe_gate(%sw_d#0, %sw_c#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<1>) -> (!dataflow.bits<32>, !dataflow.bits<1>)

    // Output switches
    %sw_out_d:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_out_c:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#1
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    fabric.yield %sw_out_d#0, %sw_out_c#0 : !dataflow.bits<32>, !dataflow.bits<1>
  }
}
