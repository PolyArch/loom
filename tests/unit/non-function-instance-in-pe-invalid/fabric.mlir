module {
  fabric.spatial_sw @sw_def : (!fabric.bits<64>) -> (!fabric.bits<64>)

  fabric.spatial_pe @pe_def(%a: !fabric.bits<64>, %b: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    fabric.instance @sw_def() : () -> ()
    fabric.yield
  }

  fabric.module @invalid_pe_target(%a: !fabric.bits<64>, %b: !fabric.bits<64>,
      %ctrl: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    %pe:1 = fabric.instance @pe_def(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
    fabric.yield %pe, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
