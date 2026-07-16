// RUN: not loom --split-input-file --loom-elaborate-fabric-instances %s 2>&1 | FileCheck %s

fabric.module @producer(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.yield %arg : !fabric.bits<8>
}

fabric.module @consumer(%arg : !fabric.bits<8>) -> (!fabric.bits<16>) {
  fabric.switch @WIDE [spatial]
      (!fabric.bits<16>) -> (!fabric.bits<16>)
      [{connectivity_table = ["1"]}]
  %result = fabric.instantiate @WIDE(
      %arg : !fabric.bits<8> to !fabric.bits<16>) -> (!fabric.bits<16>)
  fabric.yield %result : !fabric.bits<16>
}

fabric.module @producer_first(%arg : !fabric.bits<16>) -> (!fabric.bits<16>) {
  %produced = fabric.instantiate @producer(
      %arg : !fabric.bits<16> to !fabric.bits<8>) -> (!fabric.bits<8>)
  // CHECK: error: cannot inline fabric.module @consumer at fabric.instantiate input #0
  // CHECK-SAME: intermediate payload width 8 is narrower than source width 16 and destination width 16
  %consumed = fabric.instantiate @consumer(
      %produced : !fabric.bits<8>) -> (!fabric.bits<16>)
  fabric.yield %consumed : !fabric.bits<16>
}

// -----

fabric.module @producer(%arg : !fabric.bits<8>) -> (!fabric.bits<8>) {
  fabric.yield %arg : !fabric.bits<8>
}

fabric.module @consumer(%arg : !fabric.bits<8>) -> (!fabric.bits<16>) {
  fabric.switch @WIDE [spatial]
      (!fabric.bits<16>) -> (!fabric.bits<16>)
      [{connectivity_table = ["1"]}]
  %result = fabric.instantiate @WIDE(
      %arg : !fabric.bits<8> to !fabric.bits<16>) -> (!fabric.bits<16>)
  fabric.yield %result : !fabric.bits<16>
}

fabric.module @consumer_first(%arg : !fabric.bits<16>) -> (!fabric.bits<16>) {
  // CHECK: error: cannot inline fabric.module @consumer at fabric.instantiate input #0
  // CHECK-SAME: intermediate payload width 8 is narrower than source width 16 and destination width 16
  %consumed = fabric.instantiate @consumer(
      %produced : !fabric.bits<8>) -> (!fabric.bits<16>)
  %produced = fabric.instantiate @producer(
      %arg : !fabric.bits<16> to !fabric.bits<8>) -> (!fabric.bits<8>)
  fabric.yield %consumed : !fabric.bits<16>
}
