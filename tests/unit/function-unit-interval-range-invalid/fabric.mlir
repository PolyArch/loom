module {
  fabric.spatial_pe @add_pe(%a: i32, %b: i32) -> (i32) {
    fabric.function_unit @fu_add(%arg0: i32, %arg1: i32) -> (i32)
        [latency = 1, interval = 0] {
      %0 = arith.addi %arg0, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.yield
  }

  fabric.module @function_unit_interval_range_invalid(%a: i32, %b: i32) -> (i32) {
    %out = fabric.instance @add_pe(%a, %b) {sym_name = "pe_add"} : (i32, i32) -> (i32)
    fabric.yield %out#0 : i32
  }
}
