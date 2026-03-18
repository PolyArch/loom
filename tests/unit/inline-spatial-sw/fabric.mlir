module {
  fabric.module @inline_spatial_sw_test(
      %a: !fabric.bits<64>, %ctrl: !fabric.bits<64>)
      -> (!fabric.bits<64>, !fabric.bits<64>) {
    %sw:1 = fabric.spatial_sw @sw_inline [connectivity_table = ["1"]] (%a)
        : (!fabric.bits<64>) -> (!fabric.bits<64>)
    fabric.yield %sw#0, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
