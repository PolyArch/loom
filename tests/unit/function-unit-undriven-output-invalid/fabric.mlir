module {
  fabric.module @function_unit_undriven_output_invalid(%arg0: i32) -> (i32, i32) {
    fabric.function_unit @fu_bad(%x: i32) -> (i32, i32) [latency = 1, interval = 1] {
      %0 = arith.addi %x, %x : i32
      fabric.yield %0 : i32
    }
    fabric.yield %arg0, %arg0 : i32, i32
  }
}
