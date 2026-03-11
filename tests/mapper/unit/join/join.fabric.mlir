// Fabric with join PE (full consume/produce, exclusive group).

module {
  fabric.pe @pe_join(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (none) {
  ^bb0(%a: i32, %b: i32):
    %0 = handshake.join %a, %b : i32, i32
    fabric.yield %0 : none
  }

  fabric.module @join(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (none) {

    // 32-bit input switches
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

    // join PE: 2 x i32 -> none
    %pe0 = fabric.instance @pe_join(%sw0#0, %sw1#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> none

    // none-type output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : none
       -> none

    fabric.yield %sw_out#0 : none
  }
}
