// RUN: loom --adg %s

// Two switches form a physical cycle but have no route_table, so they are
// unconfigured and contribute no combinational edges. This is valid.
fabric.module @test_no_route(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %sw0:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %a, %sw1#0 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  %sw1:2 = fabric.switch [connectivity_table = [1, 1, 1, 1]] %sw0#0, %sw0#1 : !dataflow.bits<32> -> !dataflow.bits<32>, !dataflow.bits<32>
  fabric.yield %sw1#1 : !dataflow.bits<32>
}
