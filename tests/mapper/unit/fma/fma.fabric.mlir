// Fabric with 3-input fma PEs (fused multiply-add).

module {
  fabric.pe @pe_fma(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>,
                    %arg2: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: f32, %b: f32, %c: f32):
    %0 = math.fma %a, %b, %c : f32
    fabric.yield %0 : f32
  }

  fabric.module @fma(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %in2: !dataflow.bits<32>
  ) -> (!dataflow.bits<32>, !dataflow.bits<32>) {

    // 3 input switches (one per DFG input)
    %sw0:2 = fabric.switch [connectivity_table = [
        1, 1]]
        %in0
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    %sw1:2 = fabric.switch [connectivity_table = [
        1, 1]]
        %in1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    %sw2:2 = fabric.switch [connectivity_table = [
        1, 1]]
        %in2
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    // fma PEs (3 inputs -> 1 output)
    %pe0 = fabric.instance @pe_fma(%sw0#0, %sw1#0, %sw2#0)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    %pe1 = fabric.instance @pe_fma(%sw0#1, %sw1#1, %sw2#1)
        {sym_name = "pe_0_1"}
        : (!dataflow.bits<32>, !dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>

    // Output switch
    %sw_out:2 = fabric.switch [connectivity_table = [
        1, 1, 1, 1]]
        %pe0, %pe1
        : !dataflow.bits<32>
       -> !dataflow.bits<32>, !dataflow.bits<32>

    fabric.yield %sw_out#0, %sw_out#1 : !dataflow.bits<32>, !dataflow.bits<32>
  }
}
