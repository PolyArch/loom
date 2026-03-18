module {
  fabric.module @fifo_bypassable_test(%a: i32) -> (i32) {
    %out = fabric.fifo @fifo_0 [depth = 2, bypassable] (%a)
        attributes {bypassed = true}
        : (i32) -> (i32)
    fabric.yield %out : i32
  }
}
