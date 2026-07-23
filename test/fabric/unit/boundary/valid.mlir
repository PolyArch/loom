// RUN: loom %s | loom | FileCheck %s

// -----------------------------------------------------------------------------
// fabric.boundary [s2t] general form: 2 operands (data + tag) -> tagged channel.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @s2t_general
fabric.module @s2t_general(%d : !fabric.bits<32>, %t : !fabric.bits<4>) {
  // CHECK: fabric.boundary [s2t] %{{.*}}, %{{.*}} : (!fabric.bits<32>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
  %0 = fabric.boundary [s2t] %d, %t : (!fabric.bits<32>, !fabric.bits<4>) -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Incoming same-kind widths normalize at the boundary input endpoints.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @s2t_input_width_normalization
// CHECK: fabric.boundary [s2t] %{{.*}}, %{{.*}} : (!fabric.bits<32> to !fabric.bits<16>, !fabric.bits<8> to !fabric.bits<4>) -> !fabric.bits_tag<16, 4>
fabric.module @s2t_input_width_normalization(%d : !fabric.bits<32>,
                                             %t : !fabric.bits<8>) {
  %0 = fabric.boundary [s2t] %d, %t
       : (!fabric.bits<32> to !fabric.bits<16>,
          !fabric.bits<8> to !fabric.bits<4>)
      -> !fabric.bits_tag<16, 4>
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.boundary [s2t] configurable-tag form: canonical unconfigured
// capability and a representative configured projection.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @s2t_configurable_unconfigured
fabric.module @s2t_configurable_unconfigured(%d : !fabric.bits<32>) {
  // CHECK: fabric.boundary [s2t] %{{.*}} : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  %0 = fabric.boundary [s2t] %d
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// CHECK-LABEL: fabric.module @s2t_const_tag
fabric.module @s2t_const_tag(%d : !fabric.bits<32>) {
  // The IntegerAttr is a signless bit pattern; MLIR's printer renders an
  // unsigned tag such as 10 in i4 as the signed literal -6 (same bits).
  // CHECK: fabric.boundary [s2t] %{{.*}} {sw_configs = {tag = -6 : i4}} : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  %0 = fabric.boundary [s2t] %d {sw_configs = {tag = 10 : i4}}
       : !fabric.bits<32> -> !fabric.bits_tag<32, 4>
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.boundary [s2t] with bits<0> data (tag-only stream).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @s2t_bits_zero
fabric.module @s2t_bits_zero(%d : !fabric.bits<0>, %t : !fabric.bits<3>) {
  // CHECK: fabric.boundary [s2t] %{{.*}}, %{{.*}} : (!fabric.bits<0>, !fabric.bits<3>) -> !fabric.bits_tag<0, 3>
  %0 = fabric.boundary [s2t] %d, %t : (!fabric.bits<0>, !fabric.bits<3>) -> !fabric.bits_tag<0, 3>
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.boundary [t2t]: canonical unconfigured capability.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @t2t_unconfigured
fabric.module @t2t_unconfigured(%a : !fabric.bits_tag<32, 2>) {
  // CHECK: fabric.boundary [t2t] %{{.*}} {hw_params = [{lut_size = 4 : i32}]} : !fabric.bits_tag<32, 2> -> !fabric.bits_tag<32, 2>
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}]}
       : !fabric.bits_tag<32, 2> -> !fabric.bits_tag<32, 2>
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.boundary [t2t]: configured identity LUT (TW1 == TW2).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @t2t_identity
fabric.module @t2t_identity(%a : !fabric.bits_tag<32, 2>) {
  // CHECK: fabric.boundary [t2t] %{{.*}} {hw_params = [{lut_size = 4 : i32}], sw_configs = {lookup_table = [{dst_tag = 0 : i2, src_tag = 0 : i2}, {dst_tag = 1 : i2, src_tag = 1 : i2}]}} : !fabric.bits_tag<32, 2> -> !fabric.bits_tag<32, 2>
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 4 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i2, dst_tag = 0 : i2},
                                       {src_tag = 1 : i2, dst_tag = 1 : i2}]}}
       : !fabric.bits_tag<32, 2> -> !fabric.bits_tag<32, 2>
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.boundary [t2t]: tag remap, TW1 != TW2 (4 -> 8 widening).
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @t2t_widen_tag
fabric.module @t2t_widen_tag(%a : !fabric.bits_tag<32, 4>) {
  // CHECK: fabric.boundary [t2t] %{{.*}} {hw_params = {{.*}}lut_size = 8 : i32{{.*}} sw_configs = {lookup_table = {{.*}}}} : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 8>
  %0 = fabric.boundary [t2t] %a
       {hw_params = [{lut_size = 8 : i32}],
        sw_configs = {lookup_table = [{src_tag = 0 : i4, dst_tag = 1 : i8},
                                       {src_tag = 1 : i4, dst_tag = 7 : i8}]}}
       : !fabric.bits_tag<32, 4> -> !fabric.bits_tag<32, 8>
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.boundary [t2s] split form: 2 results.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @t2s_split
fabric.module @t2s_split(%a : !fabric.bits_tag<32, 4>) {
  // CHECK: %{{.*}}:2 = fabric.boundary [t2s] %{{.*}} : !fabric.bits_tag<32, 4> -> (!fabric.bits<32>, !fabric.bits<4>)
  %d, %t = fabric.boundary [t2s] %a : !fabric.bits_tag<32, 4> -> (!fabric.bits<32>, !fabric.bits<4>)
  fabric.yield
}

// -----------------------------------------------------------------------------
// fabric.boundary [t2s] drop-tag form: 1 result.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @t2s_drop_tag
fabric.module @t2s_drop_tag(%a : !fabric.bits_tag<32, 4>) {
  // CHECK: fabric.boundary [t2s] %{{.*}} : !fabric.bits_tag<32, 4> -> !fabric.bits<32>
  %d = fabric.boundary [t2s] %a : !fabric.bits_tag<32, 4> -> !fabric.bits<32>
  fabric.yield
}

// -----------------------------------------------------------------------------
// Combined: s2t -> t2t -> t2s round-trip in one module.
// -----------------------------------------------------------------------------

// CHECK-LABEL: fabric.module @boundary_pipeline
fabric.module @boundary_pipeline(%d : !fabric.bits<16>, %t : !fabric.bits<3>) {
  // CHECK: %[[TAGGED:.*]] = fabric.boundary [s2t]
  %tagged = fabric.boundary [s2t] %d, %t : (!fabric.bits<16>, !fabric.bits<3>) -> !fabric.bits_tag<16, 3>
  // CHECK: %[[REMAPPED:.*]] = fabric.boundary [t2t]
  %remapped = fabric.boundary [t2t] %tagged
              {hw_params = [{lut_size = 4 : i32}],
               sw_configs = {lookup_table = [{src_tag = 0 : i3, dst_tag = 5 : i3},
                                              {src_tag = 1 : i3, dst_tag = 2 : i3}]}}
              : !fabric.bits_tag<16, 3> -> !fabric.bits_tag<16, 3>
  // CHECK: fabric.boundary [t2s]
  %out = fabric.boundary [t2s] %remapped : !fabric.bits_tag<16, 3> -> !fabric.bits<16>
  fabric.yield
}
