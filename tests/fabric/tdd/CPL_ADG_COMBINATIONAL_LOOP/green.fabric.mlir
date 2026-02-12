// RUN: loom --adg %s

// Combinational cycle broken by fabric.fifo (sequential element).
fabric.module @test_comb_loop_ok(%a: i32) -> (i32) {
  %f = fabric.fifo [depth = 2] %sw1#0 : i32
  %sw0:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %f : i32 -> i32, i32
  %sw1:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %sw0#0, %sw0#1 : i32 -> i32, i32
  fabric.yield %sw1#1 : i32
}
