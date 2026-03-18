module {
  fabric.module @function_unit_unused_arg_invalid(%arg0: i32, %arg1: i32) -> (i32) {
    fabric.function_unit @fu_bad(%x: i32, %y: i32) -> (i32) [latency = 1, interval = 1] {
      %0 = arith.addi %x, %x : i32
      fabric.yield %0 : i32
    }
    fabric.yield %arg0 : i32
  }
}
