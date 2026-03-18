module {
  %c0 = arith.constant 0 : i64
  %a = builtin.unrealized_conversion_cast %c0 : i64 to !fabric.bits<64>

  %sw:1 = fabric.spatial_sw @sw_inline [connectivity_table = ["1"]] (%a)
      : (!fabric.bits<64>) -> (!fabric.bits<64>)
}
