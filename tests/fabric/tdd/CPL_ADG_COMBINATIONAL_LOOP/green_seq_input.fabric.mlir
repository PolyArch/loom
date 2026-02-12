// RUN: loom --adg %s

// Input 0 feeds the switch (combinational path to output 0).
// Input 1 feeds only the fifo (sequential path to output 1).
// Feedback wired to input 1 must NOT be flagged as a combinational loop.
fabric.module @split(%a: i32, %b: i32) -> (i32, i32) {
  // Use switch broadcast to duplicate %a for two switch inputs
  %bcast_a:2 = fabric.switch [connectivity_table = [1, 1]] %a : i32 -> i32, i32
  %sw:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %bcast_a#0, %bcast_a#1 : i32 -> i32, i32
  %seq = fabric.fifo [depth = 2] %b : i32
  fabric.yield %sw#0, %seq : i32, i32
}

fabric.module @top(%x: i32) -> (i32) {
  // Use switch broadcast to duplicate %x for two consumers
  %bcast_x:2 = fabric.switch [connectivity_table = [1, 1]] %x : i32 -> i32, i32
  // u's comb output #0 -> v's input #1 (sequential only) -> no comb edge.
  // v's comb output #0 -> u's input #1 (sequential only) -> no comb edge.
  %u:2 = fabric.instance @split(%bcast_x#0, %v#0) : (i32, i32) -> (i32, i32)
  // Broadcast u#0 for two consumers: v's input and module output
  %bcast_u0:2 = fabric.switch [connectivity_table = [1, 1]] %u#0 : i32 -> i32, i32
  %v:2 = fabric.instance @split(%bcast_x#1, %bcast_u0#0) : (i32, i32) -> (i32, i32)
  fabric.yield %bcast_u0#1 : i32
}
