module {
  fabric.spatial_sw @left_sw [connectivity_table = ["1"]] : (!fabric.bits<32>) -> (!fabric.bits<32>)
  fabric.spatial_sw @right_sw [connectivity_table = ["1"]] : (!fabric.bits<32>) -> (!fabric.bits<32>)

  fabric.spatial_pe @relay_pe(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>) -> (!fabric.bits<32>) {
    fabric.function_unit @fu0(%cond: i1, %arg0: i32) -> (i32) [latency = -1, interval = -1] {
      %0 = dataflow.invariant %cond, %arg0 : i1, i32 -> i32
      fabric.yield %0 : i32
    }
    fabric.yield
  }

  fabric.module @pe_transit_invalid(%in0: !fabric.bits<32>) -> (!fabric.bits<32>) {
    %left:1 = fabric.instance @left_sw(%in0) {sym_name = "sw_0_0"} : (!fabric.bits<32>) -> (!fabric.bits<32>)
    %pe:1 = fabric.instance @relay_pe(%left#0, %left#0) {sym_name = "pe_0_0"} : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    %right:1 = fabric.instance @right_sw(%pe#0) {sym_name = "sw_1_1"} : (!fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %right#0 : !fabric.bits<32>
  }
}
