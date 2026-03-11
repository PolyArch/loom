// Fabric with a single PE and no switches - mapping should fail
// because the mapper requires routing infrastructure.
module {
  fabric.pe @pe_addi(%arg0: !dataflow.bits<32>, %arg1: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%a: i32, %b: i32):
    %0 = arith.addi %a, %b : i32
    fabric.yield %0 : i32
  }

  fabric.module @no_switch(
      %in0: !dataflow.bits<32>,
      %in1: !dataflow.bits<32>,
      %ctrl: none
  ) -> (!dataflow.bits<32>, none) {
    %pe_out = fabric.instance @pe_addi(%in0, %in1)
        {sym_name = "pe_0_0"}
        : (!dataflow.bits<32>, !dataflow.bits<32>) -> !dataflow.bits<32>
    fabric.yield %pe_out, %ctrl : !dataflow.bits<32>, none
  }
}
