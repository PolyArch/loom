module {
  fabric.temporal_sw @sw_limit
      [num_route_table = 1, connectivity_table = [
          "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
          "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1",
          "1", "1", "1", "1", "1", "1", "1", "1", "1", "1", "1"
      ]]
      : (!fabric.tagged<!fabric.bits<32>, i2>)
        -> (!fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>,
            !fabric.tagged<!fabric.bits<32>, i2>)

  fabric.module @temporal_sw_port_count_invalid_test(
      %a: !fabric.bits<32>, %ctrl: !fabric.bits<1>)
      -> (!fabric.bits<32>, !fabric.bits<1>) {
    fabric.yield %a, %ctrl : !fabric.bits<32>, !fabric.bits<1>
  }
}
