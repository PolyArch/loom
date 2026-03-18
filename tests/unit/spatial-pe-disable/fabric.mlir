module {
  fabric.spatial_sw @result_mux [connectivity_table = ["11"]]
      : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)

  fabric.spatial_pe @add_pe(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.function_unit @fu_add(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.spatial_pe @sub_pe(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.function_unit @fu_sub(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.subi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @spatial_pe_disable_test(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    %add = fabric.instance @add_pe(%a, %b) {sym_name = "pe_add"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    %sub = fabric.instance @sub_pe(%a, %b) {sym_name = "pe_sub"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    %out = fabric.instance @result_mux(%add#0, %sub#0) {sym_name = "sw_0"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out#0 : !fabric.bits<32>
  }
}
