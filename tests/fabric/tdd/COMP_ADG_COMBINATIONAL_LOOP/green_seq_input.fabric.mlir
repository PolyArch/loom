// RUN: loom --adg %s

// Input 0 feeds the switch (combinational path to output 0).
// Input 1 feeds only the fifo (sequential path to output 1).
// Feedback wired to input 1 must NOT be flagged as a combinational loop.
fabric.module @split(%a: i32, %b: i32) -> (i32, i32) {
  %sw:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %a : i32 -> i32, i32
  %seq = fabric.fifo [depth = 2] %b : i32
  fabric.yield %sw#0, %seq : i32, i32
}

fabric.module @top(%x: i32) -> (i32) {
  // u's comb output #0 -> v's input #1 (sequential only) -> no comb edge.
  // v's comb output #0 -> u's input #1 (sequential only) -> no comb edge.
  %u:2 = fabric.instance @split(%x, %v#0) : (i32, i32) -> (i32, i32)
  %v:2 = fabric.instance @split(%x, %u#0) : (i32, i32) -> (i32, i32)
  fabric.yield %u#0 : i32
}
