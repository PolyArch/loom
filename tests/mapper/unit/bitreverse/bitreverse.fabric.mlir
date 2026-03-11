// Fabric with llvm.intr.bitreverse PE (zero-cost wire re-routing).

module {
  fabric.pe @pe_bitreverse(%arg0: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32):
    %0 = llvm.intr.bitreverse(%a) : (i32) -> i32
    fabric.yield %0 : i32
  }

  fabric.module @bitreverse(
      %in0: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // 32-bit input switch
    %sw0:1 = fabric.switch [connectivity_table = [
        1]]
        %in0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // bitreverse PE
    %pe0 = fabric.instance @pe_bitreverse(%sw0#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>) -> !dataflow.bits<32>

    // 32-bit output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
