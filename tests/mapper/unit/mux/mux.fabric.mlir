// Fabric with mux PE (partial-consume, exclusive group).

module {
  fabric.pe @pe_mux(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<32>,
                    %arg2: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%sel: index, %a: i32, %b: i32):
    %r = handshake.mux %sel [%a, %b] : index, i32
    fabric.yield %r : i32
  }

  fabric.module @mux(
      %sel_in: !dataflow.bits<57>,
      %data0: !dataflow.bits<32>,
      %data1: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // 57-bit switch for select routing (index = 57 bits)
    %sw_s:1 = fabric.switch [connectivity_table = [
        1]]
        %sel_in
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    // 32-bit switches for data routing
    %sw_d0:1 = fabric.switch [connectivity_table = [
        1]]
        %data0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_d1:1 = fabric.switch [connectivity_table = [
        1]]
        %data1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // mux PE
    %pe0 = fabric.instance @pe_mux(%sw_s#0, %sw_d0#0, %sw_d1#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<57>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // 32-bit output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
