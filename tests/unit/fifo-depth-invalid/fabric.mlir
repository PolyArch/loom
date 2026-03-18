module {
  fabric.module @fifo_depth_invalid_test(%a: i32) -> (i32) {
    %out = fabric.fifo @fifo_0 [depth = 0, bypassable] (%a)
        attributes {bypassed = false}
        : (i32) -> (i32)
    fabric.yield %out : i32
  }
}
