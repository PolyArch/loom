module {
  fabric.spatial_pe @dual_out_pe(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>) {
    fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
        [latency = 1, interval = 1] {
      %0 = arith.addi %x, %y : i32
      fabric.yield %0 : i32
    }
    fabric.yield
  }

  fabric.module @non_switch_broadcast_invalid(%a: !fabric.bits<32>,
                                              %b: !fabric.bits<32>)
      -> (!fabric.bits<32>, !fabric.bits<32>) {
    %pe:2 = fabric.instance @dual_out_pe(%a, %b) {sym_name = "pe_0_0"}
        : (!fabric.bits<32>, !fabric.bits<32>)
          -> (!fabric.bits<32>, !fabric.bits<32>)
    fabric.yield %pe#0, %pe#1 : !fabric.bits<32>, !fabric.bits<32>
  }
}
