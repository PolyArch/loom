// RUN: not loom --adg %s 2>&1 | FileCheck %s
// CHECK: CFG_MEMORY_EMPTY_TAG_RANGE

// Invalid: second region has end_tag (4) <= start_tag (4), creating an
// empty half-open range [4,4). Address translation would never match.
fabric.module @addr_translation_empty(%ldaddr: !dataflow.bits<57>, %staddr: !dataflow.bits<57>, %stdata: !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none) {
  %lddata, %lddone, %stdone = fabric.memory
      [ldCount = 1, stCount = 1, lsqDepth = 1, numRegion = 2,
       addr_offset_table = array<i64: 1, 0, 4, 256,  1, 4, 4, 512>]
      (%ldaddr, %staddr, %stdata)
      : memref<256xi32>, (!dataflow.bits<57>, !dataflow.bits<57>, !dataflow.bits<32>) -> (!dataflow.bits<32>, none, none)
  fabric.yield %lddata, %lddone, %stdone : !dataflow.bits<32>, none, none
}
