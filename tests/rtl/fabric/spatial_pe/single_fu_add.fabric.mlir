// Test: spatial PE with a single addi function unit
module {
  fabric.spatial_pe @add_pe(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.function_unit @fu_add(%a: i32, %b: i32) -> (i32)
        [latency = 1, interval = 1] {
      %r = arith.addi %a, %b : i32
      fabric.yield %r : i32
    }
    fabric.yield
  }

  fabric.module @test_spatial_pe_single_fu_add(
      %a: !fabric.bits<32>, %b: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    %out = fabric.instance @add_pe(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out#0 : !fabric.bits<32>
  }
}
