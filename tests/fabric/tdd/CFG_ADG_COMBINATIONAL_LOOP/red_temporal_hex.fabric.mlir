// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CFG_ADG_COMBINATIONAL_LOOP

// Inline temporal switch with hex-encoded route_table creating a loop.
// Tag type is i4 (tagWidth=4), full crossbar 2x2 (popcount=4).
// Hex 0xf0 = 0b1111_0000: route bits [7:4] = all 1s, tag [3:0] = 0.
// 1x1 switch B: popcount=1, hex 0x10 = route bit [4] = 1, tag [3:0] = 0.
// Feedback path: %fb_out -> B -> %fb_back -> A -> %fb_out
fabric.module @loop(%a: !dataflow.tagged<!dataflow.bits<32>, i4>)
    -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %fb_out, %result = fabric.temporal_sw
      [num_route_table = 1, connectivity_table = [1, 1, 1, 1]]
      {route_table = ["0xf0"]}
      %a, %fb_back
      : !dataflow.tagged<!dataflow.bits<32>, i4>
     -> !dataflow.tagged<!dataflow.bits<32>, i4>,
        !dataflow.tagged<!dataflow.bits<32>, i4>
  %fb_back = fabric.temporal_sw
      [num_route_table = 1, connectivity_table = [1]]
      {route_table = ["0x10"]}
      %fb_out
      : !dataflow.tagged<!dataflow.bits<32>, i4>
     -> !dataflow.tagged<!dataflow.bits<32>, i4>
  fabric.yield %result : !dataflow.tagged<!dataflow.bits<32>, i4>
}
