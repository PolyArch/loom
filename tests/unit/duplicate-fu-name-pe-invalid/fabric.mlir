module {
  fabric.module @duplicate_fu_name_pe_invalid(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %pe:1 = fabric.spatial_pe @pe_inline inputs(%a, %b)
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>) {
      fabric.function_unit @dup(%x: i32, %y: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %x, %y : i32
        fabric.yield %sum : i32
      }
      fabric.function_unit @dup(%x: i32, %y: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %x, %y : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }
    fabric.yield %pe#0, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
