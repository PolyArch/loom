// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// Anonymous spatial fabric.switch, hw-only (no sw_configs).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_spatial_anon_hw
fabric.module @sw_spatial_anon_hw(%a : !fabric.bits<32>, %b : !fabric.bits<32>,
                                  %c : !fabric.bits<32>, %d : !fabric.bits<32>) {
  // CHECK: %{{.*}}:3 = fabric.switch [spatial]
  // CHECK-SAME: connectivity_table = ["0110", "1011", "1111"]
  %o:3 = fabric.switch [spatial] %a, %b, %c, %d
         [{connectivity_table = ["0110", "1011", "1111"]}]
         : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous spatial fabric.switch, programmed (route_table + switch_enable).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_spatial_anon_prog
fabric.module @sw_spatial_anon_prog(%a : !fabric.bits<32>, %b : !fabric.bits<32>,
                                    %c : !fabric.bits<32>, %d : !fabric.bits<32>) {
  // CHECK: fabric.switch [spatial]
  // CHECK-SAME: route_table = ["01", "100", "0100"]
  // CHECK-SAME: switch_enable = true
  %o:3 = fabric.switch [spatial] %a, %b, %c, %d
         [{connectivity_table = ["0110", "1011", "1111"]}]
         {route_table = ["01", "100", "0100"], switch_enable = true}
         : (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Spatial broadcast: a single input may be selected by multiple outputs.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_spatial_broadcast
fabric.module @sw_spatial_broadcast(%a : !fabric.bits<8>, %b : !fabric.bits<8>) {
  // CHECK: fabric.switch [spatial]
  %o:3 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11", "11"]}]
         {route_table = ["10", "10", "10"], switch_enable = true}
         : (!fabric.bits<8>, !fabric.bits<8>)
        -> (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Named spatial fabric.switch template (declaration only).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.switch @MySwSpatial [spatial]
fabric.switch @MySwSpatial [spatial]
       (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>, !fabric.bits<32>)
       [{connectivity_table = ["11", "11"]}]
       {route_table = ["01", "10"], switch_enable = true}

// -----------------------------------------------------------------------------
// Anonymous temporal fabric.switch, hw-only.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_temporal_anon_hw
fabric.module @sw_temporal_anon_hw(%a : !fabric.bits_tag<32, 4>,
                                   %b : !fabric.bits_tag<32, 4>,
                                   %c : !fabric.bits_tag<32, 4>,
                                   %d : !fabric.bits_tag<32, 4>) {
  // CHECK: fabric.switch [temporal]
  // CHECK-SAME: route_table_size = 8
  %o:3 = fabric.switch [temporal] %a, %b, %c, %d
         [{connectivity_table = ["0110", "1011", "1111"], route_table_size = 8 : i32}]
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>,
            !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous temporal switch with incoming same-kind width normalization.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_temporal_input_width_normalization
// CHECK: fabric.switch [temporal]
// CHECK-SAME: : (!fabric.bits_tag<32, 8> to !fabric.bits_tag<16, 4>) -> !fabric.bits_tag<16, 4>
fabric.module @sw_temporal_input_width_normalization(
    %a : !fabric.bits_tag<32, 8>) {
  %o = fabric.switch [temporal] %a
       [{connectivity_table = ["1"], route_table_size = 1 : i32}]
       : (!fabric.bits_tag<32, 8> to !fabric.bits_tag<16, 4>)
      -> !fabric.bits_tag<16, 4>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Anonymous temporal fabric.switch, programmed (route_table_size = 2).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_temporal_anon_prog
fabric.module @sw_temporal_anon_prog(%a : !fabric.bits_tag<32, 4>,
                                     %b : !fabric.bits_tag<32, 4>,
                                     %c : !fabric.bits_tag<32, 4>,
                                     %d : !fabric.bits_tag<32, 4>) {
  // CHECK: fabric.switch [temporal]
  // CHECK-SAME: switch_enable = true
  %o:3 = fabric.switch [temporal] %a, %b, %c, %d
         [{connectivity_table = ["0110", "1011", "1111"], route_table_size = 2 : i32}]
         {
           route_table = [
             {route_sel = ["01", "100", "0100"], tag = 10 : i4, valid = true},
             {route_sel = ["10", "001", "0001"], tag = 11 : i4, valid = true}
           ],
           switch_enable = true
         }
         : (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>,
            !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Degenerate temporal: route_table_size = 2 with all entries valid=false.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_temporal_degenerate
fabric.module @sw_temporal_degenerate(%a : !fabric.bits_tag<16, 3>,
                                      %b : !fabric.bits_tag<16, 3>) {
  // CHECK: fabric.switch [temporal]
  %o:2 = fabric.switch [temporal] %a, %b
         [{connectivity_table = ["11", "11"], route_table_size = 2 : i32}]
         {
           route_table = [
             {route_sel = ["00", "00"], tag = 0 : i3, valid = false},
             {route_sel = ["00", "00"], tag = 0 : i3, valid = false}
           ],
           switch_enable = true
         }
         : (!fabric.bits_tag<16, 3>, !fabric.bits_tag<16, 3>)
        -> (!fabric.bits_tag<16, 3>, !fabric.bits_tag<16, 3>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// Named temporal fabric.switch template.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.switch @MySwTemporal [temporal]
fabric.switch @MySwTemporal [temporal]
       (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
        -> (!fabric.bits_tag<32, 4>, !fabric.bits_tag<32, 4>)
       [{connectivity_table = ["11", "11"], route_table_size = 1 : i32}]
       {
         route_table = [
           {route_sel = ["10", "01"], tag = 0 : i4, valid = true}
         ],
         switch_enable = true
       }

// -----------------------------------------------------------------------------
// switch inside module body alongside pe and fifo.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @sw_with_other_ops
fabric.module @sw_with_other_ops(%a : !fabric.bits<32>, %b : !fabric.bits<32>) {
  // CHECK: fabric.switch [spatial]
  %s:2 = fabric.switch [spatial] %a, %b
         [{connectivity_table = ["11", "11"]}]
         : (!fabric.bits<32>, !fabric.bits<32>)
        -> (!fabric.bits<32>, !fabric.bits<32>)
  fabric.yield
}
