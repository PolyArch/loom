module {
  fabric.module @function_unit_passthrough_invalid(%arg0: i32) -> (i32) {
    fabric.function_unit @fu_passthrough(%x: i32) -> (i32) [latency = 1, interval = 1] {
      fabric.yield %x : i32
    }
    fabric.yield %arg0 : i32
  }
}
