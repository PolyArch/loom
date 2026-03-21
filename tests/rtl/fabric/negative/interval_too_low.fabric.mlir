// Negative test: function_unit with interval=0.
// Non-dataflow FUs require interval >= 1 (interval=1 means fully pipelined).
// interval=0 is always invalid and SVGen must reject this with an error.
module {
  fabric.function_unit @fu_bad_interval(%a: i32, %b: i32) -> (i32)
      [latency = 0, interval = 0] {
    %r = arith.addi %a, %b : i32
    fabric.yield %r : i32
  }

  fabric.spatial_pe @pe_def(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.instance @fu_bad_interval() {sym_name = "fu_add_0"} : () -> ()
    fabric.yield
  }

  fabric.module @test_interval_too_low(
      %a: !fabric.bits<32>, %b: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    %out = fabric.instance @pe_def(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out#0 : !fabric.bits<32>
  }
}
