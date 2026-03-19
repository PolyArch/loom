module {
  fabric.spatial_sw @result_mux [connectivity_table = ["11"]]
      : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)

  fabric.spatial_pe @gate_pe(%cond: !fabric.bits<1>, %value: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.function_unit @fu_gate(%fu_cond: i1, %fu_value: i32) -> (i32)
        [latency = -1, interval = -1] {
      %0, %1 = dataflow.gate %fu_value, %fu_cond : i32, i1 -> i32, i1
      fabric.yield %0 : i32
    }
    fabric.yield
  }

  fabric.spatial_pe @invariant_pe(%cond: !fabric.bits<1>, %value: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.function_unit @fu_invariant(%fu_cond: i1, %fu_value: i32) -> (i32)
        [latency = -1, interval = -1] {
      %0 = dataflow.invariant %fu_cond, %fu_value : i1, i32 -> i32
      fabric.yield %0 : i32
    }
    fabric.yield
  }

  fabric.module @dataflow_invariant_test(%cond: !fabric.bits<1>, %value: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    %gate = fabric.instance @gate_pe(%cond, %value) {sym_name = "pe_gate"}
        : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<32>)
    %inv = fabric.instance @invariant_pe(%cond, %value)
        {sym_name = "pe_invariant"}
        : (!fabric.bits<1>, !fabric.bits<32>) -> (!fabric.bits<32>)
    %out = fabric.instance @result_mux(%gate#0, %inv#0) {sym_name = "sw_0"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out#0 : !fabric.bits<32>
  }
}
