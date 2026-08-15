// RUN: loom %s > /dev/null 2>&1

fabric.module @spatial_warning_stability() {
  fabric.pe @SpatialWarning [spatial]
      (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>)
      -> (!fabric.bits<8>, !fabric.bits<8>, !fabric.bits<8>,
          !fabric.bits<8>, !fabric.bits<8>) {
  ^bb0(%p0 : !fabric.bits<8>, %p1 : !fabric.bits<8>,
       %p2 : !fabric.bits<8>, %p3 : !fabric.bits<8>):
    fabric.fu(%f0 = %p0 : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%f0, %f0)
          {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32]}}
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
    fabric.yield
  }
  fabric.yield
}

fabric.module @temporal_warning_stability() {
  fabric.pe @TemporalWarning [temporal]
      (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
       !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>)
      -> (!fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>, !fabric.bits_tag<8, 2>,
          !fabric.bits_tag<8, 2>)
      attributes {
        tag_width = 2 : i32,
        num_instruction = 1 : i32,
        fu_config_mode = #fabric.fu_config_mode<per_fu_config>,
        operand_buffer_mode = #fabric.operand_buffer_mode<per_instruction>,
        operand_buffer_size = 2 : i32
      } {
  ^bb0(%p0 : !fabric.bits<8>, %p1 : !fabric.bits<8>,
       %p2 : !fabric.bits<8>, %p3 : !fabric.bits<8>):
    fabric.fu(%f0 = %p0 : !fabric.bits<8>) -> (!fabric.bits<8>) {
      %v = fabric.op [@arith.addi] (%f0, %f0)
          {implementation_family = #fabric.implementation_family<ScalarIntegerAddSub>, hw_params = {integer_widths = [8 : i32]}}
          : (!fabric.bits<8>, !fabric.bits<8>) -> !fabric.bits<8>
      fabric.yield %v : !fabric.bits<8>
    }
    fabric.yield
  }
  fabric.yield
}
