// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_TEMPORAL_PE_LOADSTORE

// Inner fabric.pe body contains handshake.load, making it a load/store PE.
// Load/store PEs are not allowed inside temporal_pe.
fabric.temporal_pe @tpe_bad(%in: !dataflow.tagged<!dataflow.bits<32>, i4>)
  [num_register = 0, num_instruction = 2, reg_fifo_depth = 0]
  -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  fabric.pe @fu_ls(%addr: !dataflow.bits<57>, %data: !dataflow.bits<32>, %ctrl: none) -> (!dataflow.bits<32>, !dataflow.bits<57>) {
    ^bb0(%addr: index, %data: i32, %ctrl: none):
    %d, %a = handshake.load [%addr] %data, %ctrl : index, i32
    fabric.yield %d, %a : i32, index
  }
  fabric.yield
}

fabric.module @test(%a: !dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>) {
  %out = fabric.instance @tpe_bad(%a)
      : (!dataflow.tagged<!dataflow.bits<32>, i4>) -> (!dataflow.tagged<!dataflow.bits<32>, i4>)
  fabric.yield %out : !dataflow.tagged<!dataflow.bits<32>, i4>
}
