module {
  fabric.module @spatial_components_def_in_module_test(
      %a: !fabric.bits<64>, %b: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    fabric.spatial_sw @sw_def [connectivity_table = ["11", "11"]]
        : (!fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)

    fabric.spatial_pe @pe_def(%x: !fabric.bits<64>, %y: !fabric.bits<64>)
        -> (!fabric.bits<64>) {
      fabric.function_unit @fu_add(%lhs: i32, %rhs: i32) -> (i32)
          [latency = 1, interval = 1] {
        %sum = arith.addi %lhs, %rhs : i32
        fabric.yield %sum : i32
      }
      fabric.yield
    }

    %sw:2 = fabric.instance @sw_def(%a, %b) {sym_name = "sw_0"}
        : (!fabric.bits<64>, !fabric.bits<64>)
          -> (!fabric.bits<64>, !fabric.bits<64>)
    %pe:1 = fabric.instance @pe_def(%sw#0, %sw#1) {sym_name = "pe_0"}
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
    fabric.yield %pe, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
