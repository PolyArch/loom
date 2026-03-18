module {
  fabric.module @inline_name_not_def_target_invalid(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %inline:1 = fabric.spatial_pe @pe_inline inputs(%a, %b)
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>) {
      fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %x, %y : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }

    %pe:1 = fabric.instance @pe_inline(%a, %b) {sym_name = "pe_1"}
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
    fabric.yield %inline, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
