// Test: standalone function unit definition with latency=1 addi
module {
  fabric.function_unit @fu_add(%x: i32, %y: i32) -> (i32)
      [latency = 1, interval = 1] {
    %sum = arith.addi %x, %y : i32
    fabric.yield %sum : i32
  }

  fabric.spatial_pe @pe_def(%a: !fabric.bits<32>, %b: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.instance @fu_add() {sym_name = "fu_add_0"} : () -> ()
    fabric.yield
  }

  fabric.module @test_fu_add_latency1(
      %a: !fabric.bits<32>, %b: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    %out = fabric.instance @pe_def(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out#0 : !fabric.bits<32>
  }
}
