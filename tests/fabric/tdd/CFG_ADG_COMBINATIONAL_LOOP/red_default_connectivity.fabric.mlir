// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CFG_ADG_COMBINATIONAL_LOOP

// Switch A has route_table but no explicit connectivity_table.
// Default connectivity is full crossbar (all 1s), so enabled routes
// form combinational paths. The feedback through switch B creates a loop.
fabric.module @loop(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %fb_out, %result = fabric.switch {route_table = [1, 1, 1, 1]} %a, %fb_back : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %fb_back = fabric.switch {route_table = [1]} %fb_out : !dataflow.bits<32> -> !dataflow.bits<32>
  fabric.yield %result : !dataflow.bits<32>
}
