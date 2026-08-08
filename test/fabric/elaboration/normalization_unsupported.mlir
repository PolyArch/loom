// RUN: not loom --split-input-file --loom-elaborate-fabric-instances %s 2>&1 | FileCheck %s

fabric.module @input_inner(%arg : !fabric.bits<8>) -> () {
  fabric.switch @WIDE [spatial]
      (!fabric.bits<16>) -> (!fabric.bits<16>)
      [{connectivity_table = ["1"]}]
  %unused = fabric.instantiate @WIDE(
      %arg : !fabric.bits<8> to !fabric.bits<16>) -> (!fabric.bits<16>)
  fabric.yield
}

fabric.module @input_host(%arg : !fabric.bits<32>) -> () {
  // CHECK: error: cannot inline fabric.module @input_inner at fabric.instantiate input #0
  // CHECK-SAME: intermediate payload width 8 is narrower than source width 32 and destination width 16
  fabric.instantiate @input_inner(
      %arg : !fabric.bits<32> to !fabric.bits<8>) -> ()
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  fabric.yield
}

// -----

fabric.module @output_inner(%arg : !fabric.bits<32>) -> (!fabric.bits<8>) {
  fabric.yield %arg : !fabric.bits<32> to !fabric.bits<8>
}

fabric.module @output_host(%arg : !fabric.bits<32>) -> (!fabric.bits<16>) {
  fabric.switch @WIDE [spatial]
      (!fabric.bits<16>) -> (!fabric.bits<16>)
      [{connectivity_table = ["1"]}]
  %middle = fabric.instantiate @output_inner(
      %arg : !fabric.bits<32>) -> (!fabric.bits<8>)
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  // CHECK: error: cannot inline fabric.module @output_inner at fabric.instantiate output #0
  // CHECK-SAME: intermediate payload width 8 is narrower than source width 32 and destination width 16
  %wide = fabric.instantiate @WIDE(
      %middle : !fabric.bits<8> to !fabric.bits<16>) -> (!fabric.bits<16>)
  fabric.yield %wide : !fabric.bits<16>
}

// -----

fabric.module @tag_inner(%arg : !fabric.bits_tag<8, 2>) -> () {
  fabric.switch @WIDE_TAG [temporal]
      (!fabric.bits_tag<8, 4>) -> (!fabric.bits_tag<8, 4>)
      [{connectivity_table = ["1"], route_table_size = 1 : i32}]
  %unused = fabric.instantiate @WIDE_TAG(
      %arg : !fabric.bits_tag<8, 2> to !fabric.bits_tag<8, 4>)
      -> (!fabric.bits_tag<8, 4>)
  fabric.yield
}

fabric.module @tag_host(%arg : !fabric.bits_tag<8, 8>) -> () {
  // CHECK: error: cannot inline fabric.module @tag_inner at fabric.instantiate input #0
  // CHECK-SAME: intermediate tag width 2 is narrower than source width 8 and destination width 4
  fabric.instantiate @tag_inner(
      %arg : !fabric.bits_tag<8, 8> to !fabric.bits_tag<8, 2>) -> ()
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  fabric.yield
}
