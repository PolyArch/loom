// Fabric with load PE (handshake.load exclusive body).

module {
  fabric.pe @pe_load(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>,
                     %arg2: none) -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  ^bb0(%addr: index, %data: i32, %ctrl: none):
    %dataResult, %addressResult = handshake.load [%addr] %data, %ctrl : index, i32
    fabric.yield %dataResult, %addressResult : i32, index
  }

  fabric.module @load(
      %addr_in: !dataflow.bits<57>,
      %data_in: !dataflow.bits<32>,
      %ctrl_in: none
  ) -> (!dataflow.bits<32>, !dataflow.bits<57>) {

    // 57-bit switch for address
    %sw_a:1 = fabric.switch [connectivity_table = [
        1]]
        %addr_in
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    // 32-bit switch for data
    %sw_d:1 = fabric.switch [connectivity_table = [
        1]]
        %data_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // none switch for control
    %sw_c:1 = fabric.switch [connectivity_table = [
        1]]
        %ctrl_in
        : none
       -> none

    // load PE: (index, i32, none) -> (i32, index)
    %pe0:2 = fabric.instance @pe_load(%sw_a#0, %sw_d#0, %sw_c#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<57>, !dataflow.bits<32>, none)
       -> (!dataflow.bits<32>, !dataflow.bits<57>)

    // Output switches
    %sw_out_d:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_out_a:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#1
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    fabric.yield %sw_out_d#0, %sw_out_a#0 : !dataflow.bits<32>, !dataflow.bits<57>
  }
}
