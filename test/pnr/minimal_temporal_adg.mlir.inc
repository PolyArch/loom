fabric.module @minimal_temporal_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %lhs : !fabric.bits_tag<32, 4>,
                                    %rhs : !fabric.bits_tag<32, 4>,
                                    %addr : !fabric.bits_tag<32, 4>,
                                    %ctrl : !fabric.bits_tag<0, 4>) attributes {coordinates_semantic = false, visual_layout = [{node = "mem", x = 0 : i32, y = 0 : i32}, {node = "pe", x = 1 : i32, y = 0 : i32}, {node = "switch", x = 2 : i32, y = 0 : i32}]} {
  fabric.pe [temporal] (%pa = %transport_fanout0_out0 : !fabric.bits_tag<32, 4>,
                    %pb = %transport_fanout1_out0 : !fabric.bits_tag<32, 4>) -> !fabric.bits_tag<32, 4> attributes {
         tag_width = 4 : i32,
         num_instruction = 1 : i32,
         fu_config_mode = #fabric.fu_config_mode<per_fu_config>,
         operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
         operand_buffer_size = 2 : i32
       } {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addi] (%fa, %fb) {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32, 16 : i32, 32 : i32, 64 : i32]}} : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  %sw0_out0, %sw0_out1 = fabric.switch [temporal] %transport_fanout0_out1, %transport_fanout1_out1
         [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  %mem0_data0, %mem0_done0 = fabric.mem [temporal] mgr(%mgr) load(%addr, %ctrl)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32, data_width = 32 : i32, tag_width = 4 : i32, operation_table_size = 1 : i32, dispatch_eligibility = {operation_port_requests = [[0 : i32]], subordinate_requests = []}}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<0, 4>)
  %transport_fanout0_out0, %transport_fanout0_out1 = fabric.switch [temporal] %lhs
         [{connectivity_table = ["1", "1"], route_table_size = 16 : i32}]
         : (!fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  %transport_fanout1_out0, %transport_fanout1_out1 = fabric.switch [temporal] %rhs
         [{connectivity_table = ["1", "1"], route_table_size = 16 : i32}]
         : (!fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}
