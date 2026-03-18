module {
  fabric.spatial_pe @pe_def(%a: !fabric.bits<64>, %b: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    %0 = fabric.add_tag %a {tag = 0 : i64}
        : !fabric.bits<64> -> !fabric.tagged<!fabric.bits<64>, i1>
    fabric.yield
  }

  fabric.module @invalid_tag_parent(%a: !fabric.bits<64>, %b: !fabric.bits<64>,
      %ctrl: !fabric.bits<64>) -> (!fabric.bits<64>, !fabric.bits<64>) {
    %pe:1 = fabric.instance @pe_def(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<64>, !fabric.bits<64>) -> (!fabric.bits<64>)
    fabric.yield %pe, %ctrl : !fabric.bits<64>, !fabric.bits<64>
  }
}
