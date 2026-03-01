// RUN: loom --adg %s

// A valid dataflow PE with exactly one dataflow operation (dataflow.invariant).
fabric.module @test(%d: !dataflow.bits<1>, %a: !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  %o = fabric.pe %d, %a : (!dataflow.bits<1>, !dataflow.bits<32>) -> (!dataflow.bits<32>) {
  ^bb0(%bd: i1, %ba: i32):
    %inv = dataflow.invariant %bd, %ba : i1, i32 -> i32
    fabric.yield %inv : i32
  }
  fabric.yield %o : !dataflow.bits<32>
}
