module {
  fabric.spatial_sw @sw_limit
      [connectivity_table = ["111111111111111111111111111111111"]]
      : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
         !fabric.bits<32>) -> (!fabric.bits<32>)

  fabric.module @spatial_sw_port_count_invalid_test(
      %a: !fabric.bits<32>, %ctrl: !fabric.bits<1>)
      -> (!fabric.bits<32>, !fabric.bits<1>) {
    %sw:1 = fabric.instance @sw_limit(
        %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
        %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a,
        %a, %a, %a, %a, %a, %a, %a, %a, %a, %a, %a)
        {sym_name = "sw_0"}
        : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>,
           !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %sw, %ctrl : !fabric.bits<32>, !fabric.bits<1>
  }
}
