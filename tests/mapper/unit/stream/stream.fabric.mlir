// Fabric with dataflow.stream PE (exclusive dataflow body, cont_cond_sel config).

module {
  fabric.pe @pe_stream(%arg0: !dataflow.bits<57>, %arg1: !dataflow.bits<57>,
                       %arg2: !dataflow.bits<57>)
      -> (!dataflow.bits<57>, !dataflow.bits<1>) {
  ^bb0(%start: index, %step: index, %bound: index):
    %idx, %wc = dataflow.stream %start, %step, %bound {step_op = "+=", stop_cond = "!="}
    fabric.yield %idx, %wc : index, i1
  }

  fabric.module @stream(
      %start_in: !dataflow.bits<57>,
      %step_in: !dataflow.bits<57>,
      %bound_in: !dataflow.bits<57>
  ) -> (!dataflow.bits<57>, !dataflow.bits<1>) {

    // 57-bit input switches (index type = ADDR_BIT_WIDTH = 57)
    %sw_s:1 = fabric.switch [connectivity_table = [
        1]]
        %start_in
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    %sw_st:1 = fabric.switch [connectivity_table = [
        1]]
        %step_in
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    %sw_b:1 = fabric.switch [connectivity_table = [
        1]]
        %bound_in
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    // stream PE: 3 x index -> (index, i1)
    %pe0:2 = fabric.instance @pe_stream(%sw_s#0, %sw_st#0, %sw_b#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<57>)
       -> (!dataflow.bits<57>, !dataflow.bits<1>)

    // Output switches
    %sw_out_idx:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#0
        : !dataflow.bits<57>
       -> !dataflow.bits<57>

    %sw_out_wc:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0#1
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    fabric.yield %sw_out_idx#0, %sw_out_wc#0 : !dataflow.bits<57>, !dataflow.bits<1>
  }
}
