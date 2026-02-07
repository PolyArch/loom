// RUN: loom --adg %s

// A module with combinational output 0 (switch -> yield) and sequential
// output 1 (switch -> fifo -> yield). Feedback that only uses the sequential
// output (result #1) must NOT be flagged as a combinational loop.
fabric.module @mixed(%a: i32, %b: i32) -> (i32, i32) {
  %sw:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %b : i32 -> i32, i32
  %seq = fabric.fifo [depth = 2] %sw#1 : i32
  fabric.yield %sw#0, %seq : i32, i32
}

fabric.module @top(%x: i32) -> (i32) {
  %u:2 = fabric.instance @mixed(%x, %v#1) : (i32, i32) -> (i32, i32)
  %v:2 = fabric.instance @mixed(%x, %u#1) : (i32, i32) -> (i32, i32)
  fabric.yield %u#0 : i32
}
