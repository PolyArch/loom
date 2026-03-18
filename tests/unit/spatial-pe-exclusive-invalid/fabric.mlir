module {
  fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
      [latency = 1, interval = 1] {
    %sum = arith.addi %x, %y : i32
    fabric.yield %sum : i32
  }

  fabric.function_unit @fu_mul(%x: i32, %y: i32) -> (i32)
      [latency = 1, interval = 1] {
    %prod = arith.muli %x, %y : i32
    fabric.yield %prod : i32
  }

  fabric.spatial_pe @pe_def(%a: !fabric.bits<32>, %b: !fabric.bits<32>,
                             %c: !fabric.bits<32>, %d: !fabric.bits<32>,
                             %ctrl: !fabric.bits<1>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<1>) {
    fabric.instance @fu_add() {sym_name = "fu_add_0"} : () -> ()
    fabric.instance @fu_mul() {sym_name = "fu_mul_0"} : () -> ()
    fabric.yield
  }

  fabric.module @spatial_pe_exclusive_invalid_test(
      %a: !fabric.bits<32>, %b: !fabric.bits<32>, %c: !fabric.bits<32>,
      %d: !fabric.bits<32>, %ctrl: !fabric.bits<1>)
      -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<1>) {
    %pe:3 = fabric.instance @pe_def(%a, %b, %c, %d, %ctrl)
        {sym_name = "pe_0"}
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<1>)
          -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<1>)
    fabric.yield %pe#0, %pe#1, %pe#2
        : !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<1>
  }
}
