module {
  fabric.spatial_pe @pe_def(%a: !fabric.bits<64>, %b: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
        [latency = 1, interval = 1] {
      %sum = arith.addi %x, %y : i32
      fabric.yield %sum : i32
    }
    fabric.yield
  }

  %c0 = arith.constant 0 : i64
  %a = builtin.unrealized_conversion_cast %c0 : i64 to !fabric.bits<64>
  %b = builtin.unrealized_conversion_cast %c0 : i64 to !fabric.bits<64>
  %pe:1 = fabric.instance @pe_def(%a, %b)
      : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
}
