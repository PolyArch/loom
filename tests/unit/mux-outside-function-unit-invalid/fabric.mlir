module {
  fabric.module @invalid_mux_parent(%a: !fabric.bits<64>, %b: !fabric.bits<64>)
      -> (!fabric.bits<64>) {
    %0 = fabric.mux %a, %b {sel = 0 : i64, discard = false, disconnect = false}
        : !fabric.bits<64>, !fabric.bits<64> -> !fabric.bits<64>
    fabric.yield %0 : !fabric.bits<64>
  }
}
