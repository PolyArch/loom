// Fabric with constant PE (handshake.constant exclusive body).

module {
  fabric.pe @pe_constant(%arg0: none) -> (!dataflow.bits<32>) {
  ^bb0(%ctrl: none):
    %c = handshake.constant %ctrl {value = 0 : i32} : i32
    fabric.yield %c : i32
  }

  fabric.module @constant(
      %ctrl_in: none
  ) -> (!dataflow.bits<32>) {

    // Constant PE: control input directly, data output through switch
    %pe0 = fabric.instance @pe_constant(%ctrl_in)
        {sym_name = "pe_0_0"}
        : (none) -> !dataflow.bits<32>

    // 32-bit output switch
    %sw0:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw0#0 : !dataflow.bits<32>
  }
}
