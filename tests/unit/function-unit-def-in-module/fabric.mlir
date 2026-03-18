module {
  fabric.module @function_unit_def_in_module_test(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
        [latency = 1, interval = 1] {
      %sum = arith.addi %x, %y : i32
      fabric.yield %sum : i32
    }

    %pe:1 = fabric.spatial_pe @pe_inline inputs(%a, %b)
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>) {
      fabric.instance @fu_add() {sym_name = "fu_add_0"} : () -> ()
      fabric.yield
    }

    fabric.yield %pe, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
