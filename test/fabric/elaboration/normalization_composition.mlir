// RUN: loom --loom-elaborate-fabric-instances %s | FileCheck %s
// RUN: loom --loom-elaborate-fabric-instances %s | loom | FileCheck %s --check-prefix=ROUNDTRIP

fabric.module @inner(%arg : !fabric.bits<16>) -> (!fabric.bits<16>) {
  fabric.switch @NARROW [spatial]
      (!fabric.bits<8>) -> (!fabric.bits<8>)
      [{connectivity_table = ["1"]}]
  %narrow = fabric.instantiate @NARROW(
      %arg : !fabric.bits<16> to !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %narrow : !fabric.bits<8> to !fabric.bits<16>
}

// CHECK-LABEL: fabric.module @host
// CHECK: %[[NARROW:.*]] = fabric.switch [spatial]
// CHECK-SAME: !fabric.bits<32> to !fabric.bits<8>
// CHECK: %[[WIDE:.*]] = fabric.switch [spatial]
// CHECK-SAME: %[[NARROW]]
// CHECK-SAME: !fabric.bits<8> to !fabric.bits<32>
// CHECK: fabric.yield %[[WIDE]] : !fabric.bits<32>
// ROUNDTRIP-LABEL: fabric.module @host
// ROUNDTRIP: fabric.switch [spatial]
// ROUNDTRIP-SAME: !fabric.bits<32> to !fabric.bits<8>
// ROUNDTRIP: fabric.switch [spatial]
// ROUNDTRIP-SAME: !fabric.bits<8> to !fabric.bits<32>
fabric.module @host(%arg : !fabric.bits<32>) -> (!fabric.bits<32>) {
  fabric.switch @WIDE [spatial]
      (!fabric.bits<32>) -> (!fabric.bits<32>)
      [{connectivity_table = ["1"]}]
  %middle = fabric.instantiate @inner(
      %arg : !fabric.bits<32> to !fabric.bits<16>) -> (!fabric.bits<16>)
      {domain_slot_bindings = array<i64: 0, 0, 0, 1, 0, 0>}
  %wide = fabric.instantiate @WIDE(
      %middle : !fabric.bits<16> to !fabric.bits<32>) -> (!fabric.bits<32>)
  fabric.yield %wide : !fabric.bits<32>
}
