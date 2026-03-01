// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CPL_PE_LOADSTORE_TAG_MODE

// Valid handshake.load body, but invalid attribute combo: both output_tag
// (TagOverwrite mode) and lqDepth (TagTransparent mode) are present.
fabric.module @test(
    %addr: !dataflow.tagged<!dataflow.bits<57>, i4>,
    %data: !dataflow.tagged<!dataflow.bits<32>, i4>,
    %ctrl: !dataflow.tagged<none, i4>
) -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<57>, i4>) {
  %d, %a = fabric.pe %addr, %data, %ctrl
      [lqDepth = 4]
      {output_tag = [0 : i4, 0 : i4]}
      : (!dataflow.tagged<!dataflow.bits<57>, i4>, !dataflow.tagged<!dataflow.bits<32>, i4>,
         !dataflow.tagged<none, i4>)
      -> (!dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<57>, i4>) {
  ^bb0(%x: index, %y: i32, %c: none):
    %ld_d, %ld_a = handshake.load [%x] %y, %c : index, i32
    fabric.yield %ld_d, %ld_a : i32, index
  }
  fabric.yield %d, %a : !dataflow.tagged<!dataflow.bits<32>, i4>, !dataflow.tagged<!dataflow.bits<57>, i4>
}
