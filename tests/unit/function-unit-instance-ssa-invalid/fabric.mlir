module {
  fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
      [latency = 1, interval = 1] {
    %sum = arith.addi %x, %y : i32
    fabric.yield %sum : i32
  }

  fabric.module @function_unit_instance_ssa_invalid(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %pe:1 = fabric.spatial_pe @pe_inline inputs(%a, %b)
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>) {
      %bad:1 = fabric.instance @fu_add(%a) {sym_name = "fu_add_0"}
          : (!fabric.bits<64>) -> (!fabric.bits<64>)
      fabric.yield %bad#0 : !fabric.bits<64>
    }
    fabric.yield %pe#0, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
