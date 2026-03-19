module {
  fabric.spatial_pe @bad_pe(%a: i32, %b: i32) -> (i32) {
    fabric.function_unit @fu_bad(%arg0: !fabric.bits<32>, %arg1: i32) -> (i32)
        [latency = 1, interval = 1] {
      %0 = arith.addi %arg1, %arg1 : i32
      fabric.yield %0 : i32
    }
    fabric.yield
  }

  fabric.module @function_unit_port_fabric_type_invalid(%a: i32, %b: i32)
      -> (i32) {
    %out = fabric.instance @bad_pe(%a, %b) {sym_name = "pe_bad"}
        : (i32, i32) -> (i32)
    fabric.yield %out#0 : i32
  }
}
