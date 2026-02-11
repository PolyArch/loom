// RUN: loom --adg %s

// Two switches with diagonal connectivity (out0<-in0, out1<-in1 only).
// Crossed feedback wires: sw0 out0 -> sw1 in1, sw1 out0 -> sw0 in1.
// Despite forming a connection cycle, no zero-delay loop exists because
// the switches cannot route in1 -> out0 (connectivity [1,0,0,1]).
fabric.module @top(%x: i32) -> (i32) {
  // Use switch broadcast to duplicate %x for two consumers
  %bcast:2 = fabric.switch [connectivity_table = [1, 1]] %x : i32 -> i32, i32
  %sw0:2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] %bcast#0, %sw1#0 : i32 -> i32, i32
  %sw1:2 = fabric.switch [connectivity_table = [1, 0, 0, 1]] %bcast#1, %sw0#0 : i32 -> i32, i32
  fabric.yield %sw0#1 : i32
}
