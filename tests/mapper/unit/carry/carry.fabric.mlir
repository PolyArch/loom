// Fabric with dataflow.carry PE (exclusive dataflow body).

module {
  fabric.pe @pe_carry(%arg0: !dataflow.bits<1>, %arg1: !dataflow.bits<32>,
                      %arg2: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%ac: i1, %init: i32, %next: i32):
    %val = dataflow.carry %ac, %init, %next : i1, i32, i32 -> i32
    fabric.yield %val : i32
  }

  fabric.module @carry(
      %ctrl_in: !dataflow.bits<1>,
      %init_in: !dataflow.bits<32>,
      %next_in: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

    // 1-bit switch for activate condition
    %sw_c:1 = fabric.switch [connectivity_table = [
        1]]
        %ctrl_in
        : !dataflow.bits<1>
       -> !dataflow.bits<1>

    // 32-bit switches for init and next values
    %sw_i:1 = fabric.switch [connectivity_table = [
        1]]
        %init_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    %sw_n:1 = fabric.switch [connectivity_table = [
        1]]
        %next_in
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    // carry PE
    %pe0 = fabric.instance @pe_carry(%sw_c#0, %sw_i#0, %sw_n#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<1>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // 32-bit output switch
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %pe0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
