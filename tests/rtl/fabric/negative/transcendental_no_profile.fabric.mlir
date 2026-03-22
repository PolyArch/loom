// Negative test: function_unit containing math.sin without --fp-ip-profile.
// Transcendental FP operations require a vendor IP profile; SVGen must
// reject this when no profile is provided.
module {
  fabric.function_unit @fu_sin(%x: f32) -> (f32)
      [latency = 10, interval = 1] {
    %r = math.sin %x : f32
    fabric.yield %r : f32
  }

  fabric.spatial_pe @pe_def(%p0: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    fabric.instance @fu_sin() {sym_name = "fu_sin_0"} : () -> ()
    fabric.yield
  }

  fabric.module @test_transcendental_no_profile(
      %a: !fabric.bits<32>)
      -> (!fabric.bits<32>) {
    %out = fabric.instance @pe_def(%a) {sym_name = "pe_0"}
        : (!fabric.bits<32>) -> (!fabric.bits<32>)
    fabric.yield %out#0 : !fabric.bits<32>
  }
}
