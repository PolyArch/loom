fabric.module @minimal_spatial_adg(%mgr : memref<?x!fabric.bits<32>>,
                                    %lhs : !fabric.bits<32>,
                                    %rhs : !fabric.bits<32>,
                                    %addr : !fabric.bits<32>,
                                    %ctrl : !fabric.bits<0>) attributes {coordinates_semantic = false, visual_layout = [{node = "mem", x = 0 : i32, y = 0 : i32}, {node = "pe", x = 1 : i32, y = 0 : i32}, {node = "switch", x = 2 : i32, y = 0 : i32}]} {
  fabric.pe [spatial] (%pa = %lhs : !fabric.bits<32>,
                    %pb = %rhs : !fabric.bits<32>) -> !fabric.bits<32> {
    fabric.fu(%fa = %pa : !fabric.bits<32>,
              %fb = %pb : !fabric.bits<32>) -> !fabric.bits<32> {
      %sum = fabric.op [@arith.addi] (%fa, %fb) : (!fabric.bits<32>, !fabric.bits<32>) -> !fabric.bits<32>
      fabric.yield %sum : !fabric.bits<32>
    }
  }
  %sw0_out0, %sw0_out1 = fabric.switch [spatial] %lhs, %rhs
         [{connectivity_table = ["11", "11"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  %mem0_data0, %mem0_done0 = fabric.mem [spatial] mgr(%mgr) load(%addr, %ctrl)
        [{load_group_size = 1 : i32, store_group_size = 0 : i32}]
        : (memref<?x!fabric.bits<32>>, !fabric.bits<32>, !fabric.bits<0>)
        -> (!fabric.bits<32>, !fabric.bits<0>)
  fabric.yield
}
