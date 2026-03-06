// RUN: loom --adg %s

// Named switch cycle broken by a named fifo instance.
fabric.switch @xbar [connectivity_table = [1, 1, 1, 1]] : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
fabric.fifo @buf [depth = 2] : (!dataflow.bits<32>) -> (!dataflow.bits<32>)

fabric.module @test(%a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %f = fabric.instance @buf(%sw1#0) : (!dataflow.bits<32>) -> (!dataflow.bits<32>)
  %sw0:2 = fabric.instance @xbar(%a, %f) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  %sw1:2 = fabric.instance @xbar(%sw0#0, %sw0#1) : (!dataflow.bits<32>, !dataflow.bits<32>) -> (!dataflow.bits<32>, !dataflow.bits<32>)
  fabric.yield %sw1#1 : !dataflow.bits<32>
}
