module {
  fabric.module @fifo_bypassed_invalid_test(%a: i32) -> (i32) {
    %out = fabric.fifo @fifo_0 [depth = 2] (%a) attributes {bypassed = true}
        : (i32) -> (i32)
    fabric.yield %out : i32
  }
}
