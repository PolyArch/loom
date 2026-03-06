// RUN: loom --adg %s

// A valid load PE: body contains exactly one handshake.load and no other
// non-terminator operations.
fabric.module @test(%addr: !dataflow.bits<57>, %data: !dataflow.bits<32>, %ctrl: none) -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  %d, %a = fabric.pe %addr, %data, %ctrl
      : (!dataflow.bits<57>, !dataflow.bits<32>, none) -> (!dataflow.bits<32>, !dataflow.bits<57>) {
  ^bb0(%x: index, %y: i32, %c: none):
    %ld_d, %ld_a = handshake.load [%x] %y, %c : index, i32
    fabric.yield %ld_d, %ld_a : i32, index
  }
  fabric.yield %d, %a : !dataflow.bits<32>, !dataflow.bits<57>
}
