module {
  fabric.module @duplicate_def_name_module_invalid(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.function_unit @dup(%x: i32, %y: i32) -> (i32)
        [latency = 1, interval = 1] {
      %sum = arith.addi %x, %y : i32
      fabric.yield %sum : i32
    }

    fabric.spatial_pe @dup(%a0: !fabric.bits<64>, %b0: !fabric.bits<64>)
        -> (!fabric.bits<64>) {
      fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %x, %y : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }

    %pe:1 = fabric.instance @dup(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
    fabric.yield %pe, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
