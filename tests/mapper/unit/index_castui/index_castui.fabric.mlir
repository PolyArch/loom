// Fabric with index_castui PE (index -> i32 unsigned conversion).

module {
  fabric.pe @pe_index_castui(%arg0: !dataflow.bits<57>) -> (!dataflow.bits<32>) {
  ^bb0(%a: index):
    %0 = arith.index_castui %a : index to i32
    fabric.yield %0 : i32
  }

  fabric.module @index_castui(
      %in0: !dataflow.bits<57>
  ) -> (!dataflow.bits<32>) {

    // 57-bit input switch (index type)
    %sw0:1 = fabric.switch [connectivity_table = [
        1]]
        %in0
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    // index_castui PE
    %pe0 = fabric.instance @pe_index_castui(%sw0#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<57>) -> !dataflow.bits<32>

    // 32-bit output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
