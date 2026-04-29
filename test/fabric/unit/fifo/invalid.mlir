// RUN: loom %s -split-input-file -verify-diagnostics

// -----
// max_depth must be > 0.
fabric.module @fifo_bad_depth_zero(%a : !fabric.bits<8>) {
  // expected-error @+1 {{'max_depth' must be > 0}}
  %0 = fabric.fifo %a [max_depth = 0, bypassable = false] : !fabric.bits<8>
  fabric.yield
}

// -----
// bypassed cannot be set when bypassable is false.
fabric.module @fifo_bypassed_without_bypassable(%a : !fabric.bits<8>) {
  // expected-error @+1 {{'bypassed' software parameter is only allowed when 'bypassable' is true}}
  %0 = fabric.fifo %a [max_depth = 4, bypassable = false] {bypassed = true} : !fabric.bits<8>
  fabric.yield
}

// -----
// bypassed = false also illegal when bypassable is false.
fabric.module @fifo_bypassed_false_without_bypassable(%a : !fabric.bits<8>) {
  // expected-error @+1 {{'bypassed' software parameter is only allowed when 'bypassable' is true}}
  %0 = fabric.fifo %a [max_depth = 4, bypassable = false] {bypassed = false} : !fabric.bits<8>
  fabric.yield
}
