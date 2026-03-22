// Negative test: function_unit with latency=0 containing arith.divsi.
// arith.divsi has intrinsic latency > 0 (WIDTH+2 cycles), so latency=0
// is invalid and SVGen must reject this with an error.
module {
  fabric.function_unit @fu_bad_div(%a: i32, %b: i32) -> (i32)
      [latency = 0, interval = 1] {
    %q = arith.divsi %a, %b : i32
    fabric.yield %q : i32
  }

  fabric.spatial_pe @pe_def(%p0: !fabric.bits<32>, %p1: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.instance @fu_bad_div() {sym_name = "fu_div_0"} : () -> ()
    fabric.yield
  }

  fabric.module @test_latency_too_low(
      %a: !fabric.bits<32>, %b: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    %out = fabric.instance @pe_def(%a, %b) {sym_name = "pe_0"}
        : (!fabric.bits<32>, !fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out#0 : !fabric.bits<32>
  }
}
