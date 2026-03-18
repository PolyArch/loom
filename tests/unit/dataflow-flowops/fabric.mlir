module {
  fabric.spatial_pe @stream_pe(
      %start: !fabric.bits<64>,
      %step: !fabric.bits<64>,
      %bound: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<1>) {
    fabric.function_unit @fu_stream(%fu_start: index, %fu_step: index, %fu_bound: index)
        -> (index, i1) [latency = 1, interval = 1] {
      %0, %1 = dataflow.stream %fu_start, %fu_step, %fu_bound
          {step_op = "+=", cont_cond = "<"}
          : (index, index, index) -> (index, i1)
      fabric.yield %0, %1 : index, i1
    }
    fabric.yield
  }

  fabric.spatial_pe @gate_pe(%value: !fabric.bits<64>, %cond: !fabric.bits<1>)
      -> (!fabric.bits<64>, !fabric.bits<1>) {
    fabric.function_unit @fu_gate(%fu_value: index, %fu_cond: i1) -> (index, i1)
        [latency = 1, interval = 1] {
      %0, %1 = dataflow.gate %fu_value, %fu_cond : index, i1 -> index, i1
      fabric.yield %0, %1 : index, i1
    }
    fabric.yield
  }

  fabric.spatial_pe @carry_pe(
      %cond: !fabric.bits<1>,
      %init: !fabric.bits<64>,
      %next: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    fabric.function_unit @fu_carry(%fu_cond: i1, %fu_init: index, %fu_next: index)
        -> (index) [latency = 1, interval = 1] {
      %0 = dataflow.carry %fu_cond, %fu_init, %fu_next : i1, index, index -> index
      fabric.yield %0 : index
    }
    fabric.yield
  }

  fabric.module @dataflow_flowops_test(
      %start: !fabric.bits<64>,
      %step: !fabric.bits<64>,
      %bound: !fabric.bits<64>,
      %init: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    %stream:2 = fabric.instance @stream_pe(%start, %step, %bound)
        {sym_name = "pe_stream"}
        : (!fabric.bits<64>, !fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<1>)
    %gate:2 = fabric.instance @gate_pe(%stream#0, %stream#1)
        {sym_name = "pe_gate"}
        : (!fabric.bits<64>, !fabric.bits<1>)
          -> (!fabric.bits<64>, !fabric.bits<1>)
    %carry = fabric.instance @carry_pe(%gate#1, %init, %gate#0)
        {sym_name = "pe_carry"}
        : (!fabric.bits<1>, !fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>)
    fabric.yield %carry#0 : !fabric.bits<64>
  }
}
