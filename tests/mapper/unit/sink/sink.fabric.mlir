// Fabric with sink PE (full consume/produce, zero outputs).

module {
  fabric.pe @pe_sink(%arg0: !dataflow.bits<32>) -> () {
  ^bb0(%a: i32):
    handshake.sink %a : i32
    fabric.yield
  }

  fabric.module @sink(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>) {

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

    // sink PE: consumes i32, produces nothing
    fabric.instance @pe_sink(%sw0#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>) -> ()

    // passthrough: second input routed to output
    %sw_out:1 = fabric.switch [connectivity_table = [
        1]]
        %sw1#0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>

    fabric.yield %sw_out#0 : !dataflow.bits<32>
  }
}
