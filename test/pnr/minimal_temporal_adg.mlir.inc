fabric.module @minimal_temporal_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %lhs : !fabric.bits_tag<32, 4>,
                                    %rhs : !fabric.bits_tag<32, 4>,
                                    %addr : !fabric.bits_tag<32, 4>,
                                    %ctrl : !fabric.bits_tag<0, 4>) {
  fabric.pe [temporal] (%pa = %lhs : !fabric.bits_tag<32, 4>,
                    %pb = %rhs : !fabric.bits_tag<32, 4>) -> !fabric.bits_tag<32, 4> attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = "per_fu_config",
         operand_buffer_mode = "per_instruction"
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addi] (%fa, %fb) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  %sw0_out0, %sw0_out1 = fabric.switch [temporal] %lhs, %rhs
         [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  %mem0_data0, %mem0_done0 = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32, tag_width = 4 : i32, addr_table_size = 1 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  fabric.yield
}
