// RUN: loom --adg %s

// Named switch cycle broken by a named fifo instance.
fabric.switch @xbar [connectivity_table = [1, 1, 1, 1]] : (i32, i32) -> (i32, i32)
fabric.fifo @buf [depth = 2] : (i32) -> (i32)

fabric.module @test(%a: i32) -> (i32) {
  %f = fabric.instance @buf(%sw1#0) : (i32) -> (i32)
  %sw0:2 = fabric.instance @xbar(%a, %f) : (i32, i32) -> (i32, i32)
  %sw1:2 = fabric.instance @xbar(%sw0#0, %sw0#1) : (i32, i32) -> (i32, i32)
  fabric.yield %sw1#1 : i32
}
